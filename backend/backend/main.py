from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.utils.logger import setup_logger

logger = setup_logger("App")

app = FastAPI(title="Rockfall Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
