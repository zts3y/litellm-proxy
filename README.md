# LiteLLM & GCP Gemini Agent Platform Integration

This project is a lightweight FastAPI-based web application and developer proof-of-concept demonstrating how to integrate **LiteLLM** with the **GCP Gemini Agent Platform** (formerly Vertex AI) to build conversational interfaces with streaming support.

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
3. Install dependencies from [requirements.txt](requirements.txt):
   ```bash
   pip install -r requirements.txt
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

### 3. Dynamic Model Garden Querying
The FastAPI backend in [main.py](main.py) shows an elegant method to query available Model Garden models dynamically using the local `gcloud` CLI tool:
```python
result = subprocess.run(
    ["gcloud", "ai", "model-garden", "models", "list", "--format=json"],
    capture_output=True,
    text=True,
    check=True
)
```
If the command fails (e.g., in headless deployments or environments without the SDK), the app gracefully reverts to a robust static fallback list of Vertex AI models:
- `vertex_ai/gemini-2.5-flash`
- `vertex_ai/gemini-2.5-pro`
- `vertex_ai/gemini-3.5-flash`
- `vertex_ai/gemini-3-flash-preview`
- `vertex_ai/gemini-3-pro-preview`

### 4. Code Implementation Patterns

#### Async Streaming via Server-Sent Events (SSE)
FastAPI and LiteLLM allow you to stream output token-by-token, which is critical for natural agentic conversations. The app implements this inside [main.py](main.py#L93-L118):

```python
async def event_generator(model: str, messages: List[Dict], project: str, location: str, temperature: Optional[float], max_tokens: Optional[int]):
    try:
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
                yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```
The client-side JavaScript in [app.js](static/app.js) parses this stream using the `EventSource` interface or an async `fetch` stream to update the chat bubbles in real-time.

---

## 📂 Project Structure

- 📁 [docs/](docs)
  - 📁 [intent/](docs/intent)
    - [vertex_ai_chat.md](docs/intent/vertex_ai_chat.md): Statement of intent detailing the project's purpose and scope constraints.
- 📁 [static/](static)
  - [index.html](static/index.html): HTML UI layout of the single-page chat interface.
  - [style.css](static/style.css): Vanilla CSS styling the responsive, glassmorphic UI.
  - [app.js](static/app.js): Client-side script handling SSE connection and interactive chat updates.
- [main.py](main.py): FastAPI backend script setting up endpoints for model discovery and streaming completions.
- [test_connection.py](test_connection.py): CLI utility to test GCP authentication and execute a basic Vertex AI prompt.
- [requirements.txt](requirements.txt): Python dependencies needed to run the application.
- [.env.example](.env.example): Environment variable configuration template.
