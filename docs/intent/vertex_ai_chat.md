# Statement of Intent: LiteLLM Vertex AI Chat Interface

- **Outcome:** A FastAPI Python backend serving a clean single-page HTML/JS chat interface, using the `litellm` Python library to interact with GCP Vertex AI Gemini models.
- **User:** A single developer, acting as a proof-of-concept for work.
- **Why now:** To test connectivity to Vertex AI Gemini models, try out different models, and verify streaming behavior.
- **Success:** The UI displays a model dropdown (trying to fetch/fallback gracefully), a toggle for streaming, and a chat interface that successfully streams responses word-by-word using LiteLLM + Vertex AI.
- **Constraint:** Runs locally, authenticates using active local Application Default Credentials (ADC), and loads GCP project/region details from a `.env` file.
- **Out of scope:** User authentication, saving chat history to a database, or containerized/production deployment setups.
