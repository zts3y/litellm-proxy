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
    """Query model garden using gcloud CLI and filter for Gemini models, or return fallbacks."""
    try:
        # Run gcloud command to get model-garden models in JSON
        result = subprocess.run(
            ["gcloud", "ai", "model-garden", "models", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        models_data = json.loads(result.stdout)
        gemini_models = []
        for model in models_data:
            name = model.get("name", "")
            # e.g., "publishers/google/models/gemini-2.5-flash"
            if "gemini" in name:
                model_base = name.split("/")[-1]  # "gemini-2.5-flash"
                model_id = f"vertex_ai/{model_base}"
                # Format a pretty display name
                pretty_name = model_base.replace("-", " ").title()
                if not any(m["id"] == model_id for m in gemini_models):
                    gemini_models.append({"id": model_id, "name": pretty_name})
        
        if not gemini_models:
            logger.info("No gemini models found in gcloud output, using fallback list.")
            return FALLBACK_MODELS
            
        logger.info(f"Dynamically discovered models: {[m['id'] for m in gemini_models]}")
        return gemini_models
    except Exception as e:
        logger.warning(f"Failed to query model-garden via gcloud: {e}. Using fallback list.")
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

async def event_generator(model: str, messages: List[Dict], project: str, location: str, temperature: Optional[float], max_tokens: Optional[int]):
    """Async generator yielding Server-Sent Events (SSE) chunks from LiteLLM."""
    try:
        # Call LiteLLM async stream completion
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            vertex_project=project,
            vertex_location=location,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content if delta.content is not None else ""
            if content:
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

    # Prepare messages
    formatted_messages = []
    if request.system_prompt:
        formatted_messages.append({"role": "system", "content": request.system_prompt})
    
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    if request.stream:
        return StreamingResponse(
            event_generator(
                model=request.model,
                messages=formatted_messages,
                project=project,
                location=location,
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
            response = await litellm.acompletion(
                model=request.model,
                messages=formatted_messages,
                vertex_project=project,
                vertex_location=location,
                stream=False,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return {
                "content": response.choices[0].message.content or ""
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
