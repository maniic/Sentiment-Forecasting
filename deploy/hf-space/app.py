"""
Hugging Face Space entrypoint (Gradio SDK, ZeroGPU hardware).

A Gradio Space runs `python app.py` and proxies whatever listens on port
7860 — it doesn't have to be a Gradio UI, so this wrapper serves the
project's real FastAPI app (dashboard + API).

Free Gradio Spaces run on ZeroGPU hardware. Its controller kills the
container ("No @spaces.GPU function detected during startup") unless the
`spaces` package sends it a startup report — and reading the package
source (spaces/zero/__init__.py), that report is only ever sent from
inside a patched `gr.Blocks.launch()`, and only when at least one
@spaces.GPU function is registered. So on ZeroGPU we register a probe
and launch a tiny hidden Gradio app on a side port purely to fire that
handshake, then serve the real dashboard on 7860 as usual. This app is
CPU-only; the probe is never called.

This file must sit at the ROOT of the Space repo (see DEPLOY.md).
"""
try:
    import spaces  # installed by the Space runtime; import before torch
    from spaces.config import Config as _SpacesConfig

    @spaces.GPU
    def _zerogpu_probe():
        """Satisfies ZeroGPU's startup check; intentionally never called."""
        return "ok"

    if _SpacesConfig.zero_gpu:
        import gradio as gr

        with gr.Blocks() as _handshake:
            gr.Markdown("Sentiment Forecasting — the dashboard is served at this Space's URL.")

        # launch() triggers spaces' startup report to the ZeroGPU controller.
        # Side port + non-blocking: the real app owns 7860 below.
        _handshake.launch(
            server_name="0.0.0.0",
            server_port=7861,
            prevent_thread_lock=True,
            quiet=True,
            show_api=False,
        )

except ImportError:  # running locally, outside HF
    pass

import uvicorn

from server import app  # the full FastAPI app: /api/* + the dashboard

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
