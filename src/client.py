# Google Gen AI SDK (latest)
import os

from google import genai


def make_client() -> genai.Client:
    # If GOOGLE_API_KEY is set, SDK will pick it automatically, but we pass for clarity.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment."
        )
    # Use Developer API by default (you can switch to Vertex by passing vertexai=True, project, location).
    client = genai.Client(api_key=api_key)
    return client
