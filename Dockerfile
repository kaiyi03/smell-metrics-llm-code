# Hugging Face Space (Docker SDK) / any container host for the evaluation dashboard.
# Serves the Flask app publicly with code execution OFF by default (ALLOW_EXEC=0).
FROM python:3.11-slim

WORKDIR /app

# Dashboard dependencies only -- NOT torch/transformers/datasets (those are the GPU phase).
COPY deploy/requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# WordNet for METEOR (optional; METEOR degrades to None without it, the app still runs).
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d $NLTK_DATA wordnet omw-1.4 || true

COPY . .

# Public-demo defaults: listen on all interfaces, HF's port, no code execution, no browser popup.
ENV HOST=0.0.0.0 PORT=7860 ALLOW_EXEC=0 DASH_NO_BROWSER=1
EXPOSE 7860

CMD ["python", "dashboard/app.py"]
