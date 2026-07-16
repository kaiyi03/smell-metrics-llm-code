# Putting the dashboard online (a link to share)

The dashboard is a Flask app with a Python backend, so it **cannot** run on GitHub
Pages (that's static hosting). To get a public URL, deploy it to a host that runs
Python. Two easy options — pick one.

Everything needed is already in the repo: `Dockerfile`, `deploy/requirements-space.txt`
(dashboard-only deps), and `deploy/README-space.md`. Code execution is OFF by default
(`ALLOW_EXEC=0`), so a public visitor can't run arbitrary code.

---

## Option A — Hugging Face Spaces (Docker)

1. Create a free account at <https://huggingface.co> and verify your email.
2. New Space: <https://huggingface.co/new-space>
   - Name: e.g. `code-smell-dashboard`
   - **Space SDK: Docker → Blank**
   - Visibility: **Public** (so Noor can open it without an account)
3. A Space is a git repo. Clone it next to this one:
   ```
   git clone https://huggingface.co/spaces/<your-user>/code-smell-dashboard
   ```
4. Copy the project into it, and use the HF README (it carries the Docker settings):
   ```
   cd code-smell-dashboard
   cp -r ../Code-Smells/{dashboard,eval_tool,smell_injection,Dockerfile,deploy} .
   cp ../Code-Smells/deploy/README-space.md README.md
   ```
5. Push — HF builds the image and starts it (~3–5 min):
   ```
   git add -A && git commit -m "dashboard" && git push
   ```
6. Your link: `https://huggingface.co/spaces/<your-user>/code-smell-dashboard` — share it.

**Turn correctness on later (optional):** Space → Settings → *Variables and secrets* →
add `ALLOW_EXEC = 1`. Only for a trusted audience — it executes submitted code inside
the (isolated, ephemeral) container.

---

## Option B — Render (no Docker, connects to GitHub)

1. Sign up at <https://render.com> with your GitHub account.
2. **New → Web Service**, pick the `Code-Smells` repo.
3. Settings:
   - Build command: `pip install -r deploy/requirements-space.txt`
   - Start command: `python dashboard/app.py`
   - Environment variables: `HOST=0.0.0.0`, `ALLOW_EXEC=0`, `DASH_NO_BROWSER=1`
     (Render sets `PORT` itself; the app reads it.)
4. Create — Render builds and gives you a `*.onrender.com` URL. It auto-redeploys on
   every push to `main`.

---

## Notes / limits

- **duplicate_code needs jscpd** (a Node tool), which isn't in the slim image, so that
  one smell won't fire on the hosted demo; the other 11 work. Ask me to add Node to the
  Dockerfile if you want it.
- **Free tiers sleep** after inactivity and wake on the next visit (~30–50 s cold start).
- The rendered **reports** (trust table, smell guide, Qwen evaluation) are already public
  on GitHub Pages — the hosted dashboard is the interactive companion to those.
