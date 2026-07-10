# Deploying the live demo

Two hosted flavors, depending on what you want the link to showcase:

| | Render (free) | Hugging Face Spaces (free) |
|---|---|---|
| Sentiment engine | Lexicon only | **Full FinBERT transformer** |
| RAM | 512MB — torch doesn't fit | 16GB — plenty |
| Config | `render.yaml` + `Dockerfile` (lite) | `Dockerfile.hf` (full) |
| Cold start | ~30s after idle | ~30s after idle (48h) |
| Best for | Quick always-on demo link | **Fully showcasing the project** |

Both serve the same app; the only difference is whether the FinBERT option in
Advanced → Sentiment engine is available or greyed out (the app detects it at
startup and falls back to the lexicon engine gracefully either way).

## Render (lexicon demo)

1. [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints) → connect this
   GitHub repo — it reads `render.yaml` automatically.
2. Deploys from `main` on every push. Done.

## Hugging Face Spaces (full FinBERT showcase)

1. Create the Space: [huggingface.co/new-space](https://huggingface.co/new-space) →
   name it (e.g. `sentiment-forecasting`), SDK **Docker** → **Blank**, hardware
   **CPU basic (free)**.

2. From your local clone, make a deploy branch where the full Dockerfile is *the*
   Dockerfile and the README carries the Space front matter:

   ```bash
   git checkout -b hf-space main
   cp Dockerfile.hf Dockerfile

   # HF reads this YAML block from the top of README.md
   cat > /tmp/hf-header.md <<'EOF'
   ---
   title: Sentiment Forecasting
   emoji: 📰
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   EOF
   cat README.md >> /tmp/hf-header.md && mv /tmp/hf-header.md README.md

   git add -A && git commit -m "Hugging Face Space deploy"
   ```

3. Push the branch to the Space (use a HF access token with *write* scope as the
   password — create one at huggingface.co/settings/tokens):

   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/sentiment-forecasting
   git push space hf-space:main
   ```

4. The Space builds the image (~5–10 min the first time — it bakes the FinBERT
   model in so visitors never wait for the download) and serves at
   `https://huggingface.co/spaces/<you>/sentiment-forecasting`.

To update later: merge changes into `main`, then
`git checkout hf-space && git merge main && git push space hf-space:main`.

## Anywhere else (Docker)

```bash
docker build -t sentiment-forecasting .            # lite (lexicon)
docker build -f Dockerfile.hf -t sf-full .         # full (FinBERT)
docker run -p 8000:8000 -e PORT=8000 sf-full
```

Any host with ~2GB RAM runs the full image (Fly.io, Railway, a VPS).
