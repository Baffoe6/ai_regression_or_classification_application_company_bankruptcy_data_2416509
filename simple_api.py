"""
Simple API entry point for bankruptcy prediction service.
This file provides a simple standalone API that can be run directly.
"""

import uvicorn

from src.api.main import app

# Configure uvicorn logging
uvicorn_log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}

if __name__ == "__main__":
    print("🚀 Starting Bankruptcy Prediction API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("💚 Health Check: http://localhost:8000/api/v1/health")
    print("🔗 Main API: http://localhost:8000/")
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        log_config=uvicorn_log_config,
    )
