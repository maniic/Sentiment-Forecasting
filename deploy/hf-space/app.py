"""
Hugging Face Space entrypoint (Gradio SDK).

A Gradio Space just runs `python app.py` and proxies whatever listens on
port 7860 — it doesn't have to be a Gradio UI. This wrapper serves the
project's real FastAPI app (dashboard + API), so the free Gradio tier
hosts the full experience, FinBERT included.

This file must sit at the ROOT of the Space repo (see DEPLOY.md).
"""
import uvicorn

from server import app  # the full FastAPI app: /api/* + the dashboard

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
