"""
FastAPI application factory and configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
import psutil
import os
from datetime import datetime
from typing import Optional

from ..config import Config
from ..utils import get_logger
from .routes import router
from .models import ErrorResponse

logger = get_logger(__name__)

# Global variables for application state
app_start_time = time.time()
config = None


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to add response time headers."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {response.status_code} - "
                f"Time: {process_time:.3f}s - "
                f"Path: {request.url.path}"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s - "
                f"Path: {request.url.path}"
            )
            raise


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    global config

    try:
        config = Config()
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        config = None

    # Create FastAPI app
    app = FastAPI(
        title="Bankruptcy Prediction API",
        description="Enterprise-grade API for predicting company bankruptcy using machine learning models",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add custom middleware
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Include routers
    app.include_router(router, prefix="/api/v1")

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                timestamp=datetime.now().isoformat(),
            ).dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                details={"errors": exc.errors()},
                timestamp=datetime.now().isoformat(),
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal server error occurred",
                timestamp=datetime.now().isoformat(),
            ).dict(),
        )

    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Starting Bankruptcy Prediction API v2.0.0")
        logger.info("Loading models and initializing services...")

        # Initialize any required services here
        # For example: load models, connect to databases, etc.

    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Shutting down Bankruptcy Prediction API")

        # Cleanup resources here
        # For example: close database connections, save state, etc.

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        uptime = time.time() - app_start_time
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        return {
            "service": "Bankruptcy Prediction API",
            "version": "2.0.0",
            "status": "operational",
            "uptime_seconds": round(uptime, 2),
            "memory_usage_mb": round(memory_usage, 2),
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
                "health": "/api/v1/health",
                "predict": "/api/v1/predict",
                "models": "/api/v1/models",
            },
            "timestamp": datetime.now().isoformat(),
        }

    return app


def get_application() -> FastAPI:
    """Get the configured FastAPI application."""
    return create_application()


# Create the app instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )
