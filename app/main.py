from fastapi import FastAPI

app = FastAPI(title="Real Estate AI API")

@app.get("/")
def read_root():
    return {"message": "API is up and running"}
