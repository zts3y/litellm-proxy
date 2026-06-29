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
@patch("main.subprocess.run")
def test_list_gemini_models_success(mock_run):
    mock_proc = MagicMock()
    mock_proc.stdout = json.dumps([
        {"name": "publishers/google/models/gemini-2.5-flash"},
        {"name": "publishers/google/models/gemini-2.5-pro"},
        {"name": "publishers/google/models/text-bison"}
    ])
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    models = list_gemini_models()
    assert models == [
        {"id": "vertex_ai/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "vertex_ai/gemini-2.5-pro", "name": "Gemini 2.5 Pro"}
    ]
    mock_run.assert_called_once_with(
        ["gcloud", "ai", "model-garden", "models", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=True
    )

# Test list_gemini_models - Success Case but No Gemini Models
@patch("main.subprocess.run")
def test_list_gemini_models_no_gemini(mock_run):
    mock_proc = MagicMock()
    mock_proc.stdout = json.dumps([
        {"name": "publishers/google/models/text-bison"},
        {"name": "publishers/google/models/imagen"}
    ])
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    models = list_gemini_models()
    assert models == FALLBACK_MODELS

# Test list_gemini_models - Subprocess Failure Case
@patch("main.subprocess.run")
def test_list_gemini_models_failure(mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(1, "gcloud")

    models = list_gemini_models()
    assert models == FALLBACK_MODELS

# Test list_gemini_models - Subprocess Not Found / Other Exception
@patch("main.subprocess.run")
def test_list_gemini_models_exception(mock_run):
    mock_run.side_effect = FileNotFoundError("gcloud not found")

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
@patch("main.litellm.acompletion", new_callable=AsyncMock)
def test_chat_endpoint_non_streaming_success(mock_acompletion):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Hello! I am Gemini."))
    ]
    mock_acompletion.return_value = mock_response

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

    # Verify litellm.acompletion was called with correct parameters
    mock_acompletion.assert_called_once_with(
        model="vertex_ai/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        vertex_project="test-project",
        vertex_location="us-central1",
        stream=False,
        temperature=0.7,
        max_tokens=100
    )

# Test /api/chat POST Endpoint - Non-streaming failure
@patch("main.litellm.acompletion", new_callable=AsyncMock)
def test_chat_endpoint_non_streaming_failure(mock_acompletion):
    mock_acompletion.side_effect = Exception("Vertex API connection error")

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
@patch("main.litellm.acompletion", new_callable=AsyncMock)
def test_chat_endpoint_streaming_success(mock_acompletion):
    async def mock_generator():
        # Yield first chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        yield chunk1
        # Yield second chunk
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        yield chunk2

    mock_acompletion.return_value = mock_generator()

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

    # Read output stream
    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert lines[0] == 'data: {"content": "Hello"}'
    assert lines[1] == 'data: {"content": " world"}'
    assert lines[2] == 'data: [DONE]'

    # Verify litellm.acompletion called with correct parameters
    mock_acompletion.assert_called_once_with(
        model="vertex_ai/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        vertex_project="test-project",
        vertex_location="us-central1",
        stream=True,
        temperature=0.8,
        max_tokens=50
    )

# Test /api/chat POST Endpoint - Streaming failure mid-stream
@patch("main.litellm.acompletion", new_callable=AsyncMock)
def test_chat_endpoint_streaming_failure_mid_stream(mock_acompletion):
    async def mock_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Start"))]
        yield chunk
        raise ValueError("Vertex connection lost")

    mock_acompletion.return_value = mock_generator()

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
@patch("main.litellm.acompletion", new_callable=AsyncMock)
def test_chat_endpoint_streaming_failure_immediate(mock_acompletion):
    mock_acompletion.side_effect = Exception("Auth failed")

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
