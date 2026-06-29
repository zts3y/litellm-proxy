import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set env vars for tests before importing main/app
os.environ["VERTEX_PROJECT"] = "test-project"
os.environ["VERTEX_LOCATION"] = "us-central1"

from main import app, list_gemini_models, FALLBACK_MODELS

# Test list_gemini_models - Success Case with Gemini Models
@patch("main.litellm.models_by_provider")
def test_list_gemini_models_success(mock_models):
    mock_models.get.return_value = [
        "gemini-2.5-flash",
        "vertex_ai/gemini-2.5-pro",
        "text-bison"
    ]

    models = list_gemini_models()
    assert models == [
        {"id": "vertex_ai/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "vertex_ai/gemini-2.5-pro", "name": "Gemini 2.5 Pro"}
    ]
    mock_models.get.assert_called_once_with("vertex_ai", [])

# Test list_gemini_models - Success Case but No Gemini Models
@patch("main.litellm.models_by_provider")
def test_list_gemini_models_no_gemini(mock_models):
    mock_models.get.return_value = [
        "text-bison",
        "imagen"
    ]

    models = list_gemini_models()
    assert models == FALLBACK_MODELS

# Test list_gemini_models - Registry Failure Case
@patch("main.litellm.models_by_provider")
def test_list_gemini_models_failure(mock_models):
    mock_models.get.side_effect = Exception("Registry error")

    models = list_gemini_models()
    assert models == FALLBACK_MODELS


# Test /api/models GET Endpoint
def test_get_models_endpoint():
    client = TestClient(app)
    with patch("main.list_gemini_models") as mock_list:
        mock_list.return_value = [{"id": "vertex_ai/gemini-mock", "name": "Gemini Mock"}]
        response = client.get("/api/models")
        assert response.status_code == 200
        assert response.json() == [{"id": "vertex_ai/gemini-mock", "name": "Gemini Mock"}]

# Test /api/chat POST Endpoint - Missing VERTEX_PROJECT environment variable
def test_chat_endpoint_missing_project(monkeypatch):
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    client = TestClient(app)
    
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 500
    assert "GCP project ID is not configured" in response.json()["detail"]

# Test /api/chat POST Endpoint - Non-streaming success
@patch("main.Agent")
@patch("main.LiteLLMModel")
def test_chat_endpoint_non_streaming_success(mock_model_cls, mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    
    mock_result = MagicMock()
    mock_result.message = {
        "role": "assistant",
        "content": [{"text": "Hello! I am Gemini."}]
    }
    mock_agent.invoke_async = AsyncMock(return_value=mock_result)

    client = TestClient(app)
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 100
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.json() == {"content": "Hello! I am Gemini."}

    # Verify LiteLLMModel and Agent were initialized correctly
    mock_model_cls.assert_called_once_with(
        model_id="vertex_ai/gemini-2.5-flash",
        client_args={
            "vertex_project": "test-project",
            "vertex_location": "us-central1"
        },
        params={"temperature": 0.7, "max_tokens": 100}
    )
    mock_agent_cls.assert_called_once_with(
        model=mock_model_cls.return_value,
        system_prompt="You are a helpful assistant.",
        callback_handler=None
    )
    mock_agent.invoke_async.assert_called_once_with([
        {"role": "user", "content": [{"text": "Hello"}]}
    ])

# Test /api/chat POST Endpoint - Non-streaming failure
@patch("main.Agent")
@patch("main.LiteLLMModel")
def test_chat_endpoint_non_streaming_failure(mock_model_cls, mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_agent.invoke_async = AsyncMock(side_effect=Exception("Vertex API connection error"))

    client = TestClient(app)
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Vertex API connection error"

# Test /api/chat POST Endpoint - Streaming success
@patch("main.Agent")
@patch("main.LiteLLMModel")
def test_chat_endpoint_streaming_success(mock_model_cls, mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    # Strands stream_async emits text deltas via TextStreamEvent
    async def mock_generator(*args, **kwargs):
        yield {"data": "Hello"}
        yield {"data": " world"}

    mock_agent.stream_async = mock_generator

    client = TestClient(app)
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "temperature": 0.8,
        "max_tokens": 50
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Read output stream — data events are already deltas (not cumulative)
    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert lines[0] == 'data: {"content": "Hello"}'
    assert lines[1] == 'data: {"content": " world"}'
    assert lines[2] == 'data: [DONE]'

    # Verify parameters
    mock_model_cls.assert_called_once_with(
        model_id="vertex_ai/gemini-2.5-flash",
        client_args={
            "vertex_project": "test-project",
            "vertex_location": "us-central1"
        },
        params={"temperature": 0.8, "max_tokens": 50}
    )

# Test /api/chat POST Endpoint - Streaming failure mid-stream
@patch("main.Agent")
@patch("main.LiteLLMModel")
def test_chat_endpoint_streaming_failure_mid_stream(mock_model_cls, mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    async def mock_generator(*args, **kwargs):
        yield {"data": "Start"}
        raise ValueError("Vertex connection lost")

    mock_agent.stream_async = mock_generator

    client = TestClient(app)
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert lines[0] == 'data: {"content": "Start"}'
    assert lines[1] == 'data: {"error": "Vertex connection lost"}'
    assert lines[2] == 'data: [DONE]'

# Test /api/chat POST Endpoint - Streaming failure immediately on call
@patch("main.Agent")
@patch("main.LiteLLMModel")
def test_chat_endpoint_streaming_failure_immediate(mock_model_cls, mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    async def mock_generator(*args, **kwargs):
        raise Exception("Auth failed")
        yield {}  # make it a generator function

    mock_agent.stream_async = mock_generator

    client = TestClient(app)
    payload = {
        "model": "vertex_ai/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert lines[0] == 'data: {"error": "Auth failed"}'
    assert lines[1] == 'data: [DONE]'

# Test / GET Endpoint - index.html exists
@patch("main.os.path.exists")
@patch("main.FileResponse")
def test_get_root_exists(mock_fileresponse, mock_exists):
    mock_exists.return_value = True
    mock_fileresponse.return_value = MagicMock()
    
    client = TestClient(app)
    response = client.get("/")
    mock_fileresponse.assert_called_once()
    assert "static/index.html" in mock_fileresponse.call_args[0][0]

# Test / GET Endpoint - index.html does not exist
@patch("main.os.path.exists")
def test_get_root_not_exists(mock_exists):
    mock_exists.return_value = False
    
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Server is running. Frontend static/index.html not created yet."}
