import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

project = os.getenv("VERTEX_PROJECT")
location = os.getenv("VERTEX_LOCATION", "us-central1")

print(f"Using Vertex Project: {project}")
print(f"Using Vertex Location: {location}")

if not project or project == "your-gcp-project-id":
    print("Error: Please set your VERTEX_PROJECT in the .env file.")
    sys.exit(1)

try:
    import litellm
except ImportError:
    print("Error: litellm is not installed in this environment.")
    sys.exit(1)

# List of newer models to probe
candidate_models = [
    "vertex_ai/gemini-2.5-flash",
    "vertex_ai/gemini-2.5-pro",
    "vertex_ai/gemini-3.5-flash",
    "vertex_ai/gemini-3-flash-preview",
    "vertex_ai/gemini-3-pro-preview"
]

print("Attempting to connect to Vertex AI and probe newer Gemini models...")

success = False
for model in candidate_models:
    print(f"\nProbing {model}...")
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hi, write a 2-word greeting."}],
            vertex_project=project,
            vertex_location=location
        )
        print(f"-> SUCCESS with {model}!")
        print("Response:", response.choices[0].message.content)
        success = True
        break
    except Exception as e:
        print(f"-> FAILED with {model}: {e}")

if not success:
    print("\nAll models failed! Please check if your GCP account has Vertex AI enabled and if you have Model Garden permissions.")
    sys.exit(1)
