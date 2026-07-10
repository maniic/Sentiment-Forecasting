# Deploying the live demo

Two hosted flavors, depending on what you want the link to showcase:

| | Render (free) | Hugging Face Spaces (free) |
|---|---|---|
| Sentiment engine | Lexicon only | **Full FinBERT transformer** |
| RAM | 512MB — torch doesn't fit | 16GB — plenty |
| Config | `render.yaml` + `Dockerfile` (lite) | `deploy/hf-space/` (Gradio-SDK wrapper) |
| Cold start | ~30s after idle | ~30s after idle (48h) |
| Best for | Quick always-on demo link | **Fully showcasing the project** |

Both serve the same app; the only difference is whether the FinBERT option in
Advanced → Sentiment engine is available or greyed out (the app detects it at
startup and falls back to the lexicon engine gracefully either way).

## Render (lexicon demo)

1. [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints) → connect this
   GitHub repo — it reads `render.yaml` automatically.
2. Deploys from `main` on every push. Done.

## Hugging Face Spaces (full FinBERT showcase — free Gradio tier)

Docker Spaces are gated on some accounts, but the free **Gradio** SDK works fine:
a Gradio Space simply runs `python app.py` and proxies whatever listens on port
7860 — it doesn't have to be a Gradio UI. `deploy/hf-space/app.py` is a tiny
wrapper that serves this project's real FastAPI dashboard on that port.

1. Create the Space: [huggingface.co/new-space](https://huggingface.co/new-space) →
   name it (e.g. `sentiment-forecasting`), SDK **Gradio** → **Blank**, hardware
   **CPU basic (free)**.

2. From your local clone, make a deploy branch with the Space files at the root:

   ```bash
   git checkout main && git reset --hard origin/main
   git checkout -B hf-space
   cp deploy/hf-space/app.py deploy/hf-space/requirements.txt deploy/hf-space/README.md .
   rm -rf docs models output               # binaries — HF requires Xet/LFS for those
   rm -f .dockerignore Dockerfile Dockerfile.hf render.yaml   # other hosts' configs;
                                           # HF's builder honors .dockerignore, which
                                           # would hide requirements.txt from the build
   git checkout --orphan hf-space-flat     # single commit, no history (history
                                           # contains screenshots HF would reject)
   git add -A && git commit -m "Hugging Face Space deploy"
   ```

3. Push the branch to the Space (username = your HF username; password = an
   access token with *write* scope from huggingface.co/settings/tokens):

   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/sentiment-forecasting
   git push --force space hf-space-flat:main
   ```

   Afterwards, return your checkout to normal with
   `git checkout main && git reset --hard origin/main`.

4. First build takes a few minutes (torch install). The Space README's
   `preload_from_hub: yiyanghkust/finbert-tone` line makes HF download the
   FinBERT weights **during the build**, so even the very first FinBERT run
   is fast. The app serves at
   `https://huggingface.co/spaces/<you>/sentiment-forecasting`
   (direct full-screen URL: `https://<you>-sentiment-forecasting.hf.space`).

Notes from the Spaces docs: free CPU Basic = 2 vCPU / 16GB RAM; the Space
sleeps after inactivity and wakes on the next visit; outbound network is
limited to ports 80/443/8080 (fine for Google News RSS and yfinance, so
Live mode works too).

To update later: re-run the whole block above from `main` — it rebuilds the flat
deploy branch from scratch each time.

## Anywhere else (Docker)

```bash
docker build -t sentiment-forecasting .            # lite (lexicon)
docker build -f Dockerfile.hf -t sf-full .         # full (FinBERT)
docker run -p 8000:8000 -e PORT=8000 sf-full
```

Any host with ~2GB RAM runs the full image (Fly.io, Railway, a VPS).
