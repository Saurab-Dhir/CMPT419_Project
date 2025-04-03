import uvicorn
import os
import sys
from app.core.config import settings

# Ensure static directories exist
os.makedirs("static/audio", exist_ok=True)

# Verify WebSocket support
try:
    import websockets
    print("✅ websockets library is installed")
except ImportError:
    print("❌ websockets library is missing")
    print("Installing websockets library...")
    os.system(f"{sys.executable} -m pip install websockets")

if __name__ == "__main__":
    print("Starting server with WebSocket support...")
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="debug",  # More detailed logging
        ws='websockets'     # Explicitly use websockets
    ) 