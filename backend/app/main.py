from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .inference import InspectionEngine
from .routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    app.state.engine = InspectionEngine(settings)
    logging.getLogger("aerovision").info("Moteur prêt : %s", app.state.engine.info()["mode"])
    yield


app = FastAPI(
    title="AeroVision API",
    version="0.1.0",
    description="Inspection visuelle automatisée de pièces industrielles — aide à la décision, "
                "validation humaine obligatoire.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
