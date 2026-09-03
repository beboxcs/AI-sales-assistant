from fastapi import FastAPI

from app.api.v1.router import router as v1_router

app = FastAPI(title="Real Estate AI API", version="0.1.0")

app.include_router(v1_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "API is up and running"}
