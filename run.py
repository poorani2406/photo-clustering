import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload, reload_dirs=["app", "static"] if reload else None)

