"""
Hugging Face Space entrypoint (Gradio SDK, ZeroGPU hardware).

A Gradio Space runs `python app.py` and proxies whatever listens on port
7860 — it doesn't have to be a Gradio UI, so this wrapper serves the
project's real FastAPI app (dashboard + API).

Free Gradio Spaces run on ZeroGPU hardware, whose runtime refuses to start
unless at least one @spaces.GPU function is registered. This app is
CPU-only (FinBERT inference on a few dozen headlines doesn't need a GPU),
so we register a never-called probe to satisfy the check. `spaces` must be
imported before torch ever loads; keep it first.

This file must sit at the ROOT of the Space repo (see DEPLOY.md).
"""
try:
    import spaces  # installed automatically by the Space runtime

    @spaces.GPU
    def _zerogpu_probe():
        """Satisfies ZeroGPU's startup check; intentionally never called."""
        return "ok"

except ImportError:  # running locally, outside HF
    pass

import uvicorn

from server import app  # the full FastAPI app: /api/* + the dashboard

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
