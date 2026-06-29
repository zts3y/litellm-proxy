# LiteLLM & GCP Gemini Agent Platform Integration

This project is a lightweight FastAPI-based web application and developer proof-of-concept demonstrating how to integrate the **Strands Agents SDK** with **LiteLLM** and the **GCP Gemini Agent Platform** (formerly Vertex AI) to build conversational interfaces with streaming support.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+** installed.
- **Google Cloud CLI (`gcloud`)** installed and authenticated.
- A Google Cloud Project with the **Vertex AI API** (`aiplatform.googleapis.com`) enabled.
- Your user identity or service account granted the **Vertex AI User** (`roles/aiplatform.user`) IAM role.

### 2. Authentication Setup
For local development, authenticate your machine using Google Application Default Credentials (ADC):
```bash
gcloud auth application-default login
```
This generates credentials stored locally that `litellm` and `google-auth` will automatically discover and use.

### 3. Repository Installation
1. Clone or navigate to the workspace.
2. Initialize and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies from [requirements.txt](requirements.txt). Note that if you are using Python 3.14+, you must use the `--ignore-requires-python` flag (since some packages like `litellm` currently specify `<3.14` support):
   ```bash
   pip install --ignore-requires-python -r requirements.txt
   ```

### 4. Configuration
Create a `.env` file in the root directory (you can copy the template from [.env.example](.env.example)):
```bash
cp .env.example .env
```
Edit your `.env` file to specify your GCP project details:
```env
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
```

### 5. Verify Connection
Run the connection verification script to probe the available models:
```bash
python test_connection.py
```

### 6. Run the Chat Application
Start the FastAPI server:
```bash
python main.py
```
Open your web browser and navigate to `http://127.0.0.1:8000` to interact with the chat UI.

---

## 🧠 Key Learnings: LiteLLM & GCP Gemini Agent Platform

### 1. Vertex AI vs. Google AI Studio (Gemini API)
When integrating Google's Gemini models, it is crucial to understand which API provider you are using, as LiteLLM handles them differently:

| Feature | Vertex AI (GCP Gemini Agent Platform) | Google AI Studio (Gemini API) |
|---|---|---|
| **LiteLLM Prefix** | `vertex_ai/<model_name>` (e.g., `vertex_ai/gemini-2.5-flash`) | `gemini/<model_name>` (e.g., `gemini/gemini-2.5-flash`) |
| **Authentication** | IAM-based / Application Default Credentials (ADC) | Simple API Key (`GEMINI_API_KEY`) |
| **Pricing / Scoping** | Tied to GCP Project Billing and quota metrics | Tiered API key-based plans |
| **Governance** | Enterprise-grade SLA, IAM policies, and VPC service controls | Developer-focused, lower overhead |
| **Default Fallback** | LiteLLM defaults to Vertex AI if no prefix is supplied | Requires explicit `gemini/` prefix |

### 2. The Gemini Enterprise Agent Platform
Google Cloud's **Gemini Enterprise Agent Platform** is the evolution of Vertex AI, bringing an agentic architecture to enterprise workflows. 
- **Model Garden**: Google's repository of foundational and third-party models. The platform allows you to use models from Google (e.g., Gemini 2.5 Flash, 2.5 Pro, 3.5 Flash) alongside curated open-source or third-party models (like Claude) as first-class citizens.
- **LiteLLM Proxy Role**: Using LiteLLM as an orchestration or proxy layer allows you to translate standard OpenAI-compatible API calls (like completions and chat completions) directly into GCP Vertex AI payloads. This facilitates multi-model fallbacks, standardizes metrics/logging, and centralizes billing governance.

### 3. Dynamic Model Discovery via LiteLLM Registry
The FastAPI backend in [main.py](main.py) queries available Gemini models dynamically from LiteLLM's built-in provider registry:
```python
def list_gemini_models():
    vertex_models = litellm.models_by_provider.get("vertex_ai", [])
    gemini_models = [m for m in vertex_models if "gemini" in m.lower()]
    ...
```
If the registry returns no Gemini models, the app gracefully reverts to a static fallback list:
- `vertex_ai/gemini-2.5-flash`
- `vertex_ai/gemini-2.5-pro`
- `vertex_ai/gemini-3.5-flash`
- `vertex_ai/gemini-3-flash-preview`
- `vertex_ai/gemini-3-pro-preview`

### 4. Code Implementation Patterns

#### Strands Agent with Async Streaming via SSE
The app uses the **Strands Agents SDK** as an orchestration layer over LiteLLM. The `LiteLLMModel` provider routes requests through LiteLLM to Vertex AI, while the `Agent` class manages the conversation loop and streams text deltas via `stream_async()`. The app implements this inside [main.py](main.py#L88-L133):

```python
from strands import Agent
from strands.models.litellm import LiteLLMModel

async def event_generator(model_id, messages, project, location, ...):
    model = LiteLLMModel(
        model_id=model_id,
        client_args={"vertex_project": project, "vertex_location": location},
        params={"temperature": temperature, "max_tokens": max_tokens}
    )
    agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)

    async for event in agent.stream_async(messages):
        if "data" in event:
            yield f"data: {json.dumps({'content': event['data']})}\n\n"
```

Key details:
- **`callback_handler=None`** suppresses the default `PrintingCallbackHandler` that would duplicate output to stdout.
- **`stream_async`** yields `TextStreamEvent` dicts where `"data"` contains **text deltas** (not cumulative text).
- **`invoke_async`** is used for the non-streaming path, returning a full `AgentResult`.
- **Parallel tool execution** is supported via the default `ConcurrentToolExecutor`, though no tools are currently registered.

The client-side JavaScript in [app.js](static/app.js) parses this SSE stream via an async `fetch` reader to update the chat bubbles in real-time.

---

## 📂 Project Structure

- 📁 [docs/](docs)
  - 📁 [intent/](docs/intent)
    - [vertex_ai_chat.md](docs/intent/vertex_ai_chat.md): Statement of intent detailing the project's purpose and scope constraints.
- 📁 [static/](static)
  - [index.html](static/index.html): HTML UI layout of the single-page chat interface.
  - [style.css](static/style.css): Vanilla CSS styling the responsive, glassmorphic UI.
  - [app.js](static/app.js): Client-side script handling SSE connection and interactive chat updates.
- [main.py](main.py): FastAPI backend using Strands Agent + LiteLLM for model discovery and streaming completions.
- [test_main.py](test_main.py): Unit tests for all endpoints, model discovery, and streaming/non-streaming paths.
- [test_connection.py](test_connection.py): CLI utility to test GCP authentication and execute a basic Vertex AI prompt.
- [requirements.txt](requirements.txt): Python dependencies (includes `strands-agents`, `litellm`, `fastapi`).
- [.env.example](.env.example): Environment variable configuration template.
