import os
import json
import logging
import subprocess
import uvicorn
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import litellm
from strands import Agent
from strands.models.litellm import LiteLLMModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="LiteLLM Vertex AI Chat Interface")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard fallback models (Vertex AI Gemini 2.5 and 3.5 series)
FALLBACK_MODELS = [
    {"id": "vertex_ai/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "vertex_ai/gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "vertex_ai/gemini-3.5-flash", "name": "Gemini 3.5 Flash"},
    {"id": "vertex_ai/gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"},
    {"id": "vertex_ai/gemini-3-pro-preview", "name": "Gemini 3 Pro Preview"}
]

def list_gemini_models():
    """Extract Gemini models supported by Vertex AI from LiteLLM's built-in models list or return fallbacks."""
    try:
        vertex_models = litellm.models_by_provider.get("vertex_ai", [])
        gemini_models = []
        for model in vertex_models:
            if "gemini" in model.lower():
                # Ensure the model ID starts with "vertex_ai/"
                model_id = model if model.startswith("vertex_ai/") else f"vertex_ai/{model}"
                # Format a pretty display name
                model_base = model.replace("vertex_ai/", "")
                pretty_name = model_base.replace("-", " ").title()
                if not any(m["id"] == model_id for m in gemini_models):
                    gemini_models.append({"id": model_id, "name": pretty_name})
        
        if not gemini_models:
            logger.info("No Gemini models found in LiteLLM registry, using fallback list.")
            return FALLBACK_MODELS
            
        logger.info(f"Dynamically discovered models from LiteLLM: {[m['id'] for m in gemini_models]}")
        return sorted(gemini_models, key=lambda x: x["id"])
    except Exception as e:
        logger.warning(f"Failed to query LiteLLM registry: {e}. Using fallback list.")
        return FALLBACK_MODELS


# Request / Response Schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@app.get("/api/models")
async def get_models():
    """Returns available Gemini models."""
    return list_gemini_models()

async def event_generator(
    model_id: str,
    messages: List[Dict],
    project: str,
    location: str,
    system_prompt: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int]
):
    """Async generator yielding Server-Sent Events (SSE) chunks from Strands Agent."""
    try:
        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Initialize the Strands LiteLLM model provider
        model = LiteLLMModel(
            model_id=model_id,
            client_args={
                "vertex_project": project,
                "vertex_location": location,
            },
            params=params if params else None
        )

        # Initialize the Strands Agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            callback_handler=None
        )

        # Run the agent async stream
        async for event in agent.stream_async(messages):
            if "data" in event:
                content = event["data"]
                # SSE format data
                yield f"data: {json.dumps({'content': content})}\n\n"
                
    except Exception as e:
        logger.error(f"Error during streaming completion: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint supporting both JSON and streaming response via SSE."""
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    
    if not project:
        raise HTTPException(
            status_code=500,
            detail="GCP project ID is not configured. Please set VERTEX_PROJECT in your .env file."
        )

    # Format history messages for Strands Agent (role and content as a list of ContentBlock dicts)
    formatted_messages = []
    for msg in request.messages:
        formatted_messages.append({
            "role": msg.role,
            "content": [{"text": msg.content}]
        })

    if request.stream:
        return StreamingResponse(
            event_generator(
                model_id=request.model,
                messages=formatted_messages,
                project=project,
                location=location,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        try:
            params = {}
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens

            # Initialize the Strands LiteLLM model provider
            model = LiteLLMModel(
                model_id=request.model,
                client_args={
                    "vertex_project": project,
                    "vertex_location": location,
                },
                params=params if params else None
            )

            # Initialize the Strands Agent
            agent = Agent(
                model=model,
                system_prompt=request.system_prompt,
                callback_handler=None
            )

            # Invoke agent asynchronously
            result = await agent.invoke_async(formatted_messages)

            # Extract response text
            content_text = ""
            if result.message and "content" in result.message:
                for block in result.message["content"]:
                    if "text" in block:
                        content_text += block["text"]

            return {
                "content": content_text
            }
        except Exception as e:
            logger.error(f"Error during standard completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Mount static files directory and serve index.html
# Create the directory if it doesn't exist
os.makedirs("static", exist_ok=True)

@app.get("/")
async def read_index():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        # Return a temporary simple HTML until we write the real one in Task 4
        return {"message": "Server is running. Frontend static/index.html not created yet."}
    return FileResponse(index_path)

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
