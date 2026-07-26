"""Main entry point for the Puzzlegrid API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("puzzlegrid.api:app", host="0.0.0.0", port=8000, reload=True)

