import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.core.config import settings
from backend.api.endpoints import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve static directory with an absolute path so it is correct regardless
# of the process working directory (local dev vs Docker).
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
os.makedirs(STATIC_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    # Validate configuration and warn about potential problems before the
    # first request arrives so issues are visible in the server logs.
    settings.validate_api_key()
    logger.info("Static files directory: %s", STATIC_DIR)
    logger.info("%s is ready.", settings.PROJECT_NAME)
    yield
    logger.info("%s is shutting down.", settings.PROJECT_NAME)


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated visualisation PNGs from /static/<filename>.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# All chat / analytics endpoints live under /api.
app.include_router(router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": f"{settings.PROJECT_NAME} is running"}


@app.get("/health")
def health_check():
    """Lightweight liveness probe used by Docker / load-balancers."""
    return {"status": "ok"}
