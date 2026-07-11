# repo-root/app.py
#
# Entry-point shim. Hugging Face Spaces (and some tooling) look for an `app`
# object at the repository root, so we simply re-export the real FastAPI app
# that lives in app/app.py. All the actual logic is in the app/ package.
from app.app import app
