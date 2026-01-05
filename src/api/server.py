from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.utils.config import ConfigLoader


def create_app() -> FastAPI:
    app = FastAPI(
        title="DeepFlow NDR Backend",
        version="0.1.0",
        description="FastAPI backend for DeepFlow Network Detection & Response.",
    )
    app.include_router(router)
    return app


app = create_app()


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    config = ConfigLoader.get_instance().system_config
    uvicorn.run(
        "src.api.server:app",
        host=config.get("api_host", "0.0.0.0"),
        port=int(config.get("api_port", 8000)),
        reload=False,
    )
