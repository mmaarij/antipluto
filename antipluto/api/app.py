"""
antipluto.api.app
===============
FastAPI application factory for the Anti-PLUTO email security prediction service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from antipluto.api.parser import parse_rfc822_email
from antipluto.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from antipluto.api.service import InferenceService

logger = logging.getLogger(__name__)


def create_app(
    model_dir: Path | str = "models",
    classifier: str = "xgboost",
    default_mode: str = "fusion",
    inference_service: Optional[InferenceService] = None,
) -> FastAPI:
    """Create and configure a production FastAPI application instance.

    Parameters
    ----------
    model_dir : Path | str
        Path to directory containing checkpoints/ and cache/.
    classifier : str
        Active classification head: 'xgboost', 'rf', 'mlp', or 'cnn'.
    default_mode : str
        Default inference representation mode: 'fusion', 'semantic', or 'stylometric'.
    inference_service : Optional[InferenceService]
        Pre-instantiated or mocked inference service (useful for testing).

    Returns
    -------
    FastAPI
    """
    model_dir_path = Path(model_dir).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inference_service is not None:
            app.state.service = inference_service
        else:
            # 1. Startup: Load models once into memory
            logger.info("Starting Anti-PLUTO API server...")
            service = InferenceService(
                model_dir=model_dir_path,
                classifier_name=classifier,
                default_mode=default_mode,
            )
            service.load_artifacts()
            app.state.service = service
        yield
        # 2. Shutdown: Clean up resources
        logger.info("Shutting down Anti-PLUTO API server...")

    app = FastAPI(
        title="Anti-PLUTO Email Security API",
        description=(
            "Production REST API for real-time phishing and generative AI detection. "
            "Ingests raw emails (JSON or RFC 5322 MIME), applies automated 5-stage cleaning "
            "(HTML stripping, reply-chain slicing, Unicode normalization, PII masking), "
            "and supports multi-mode inference across Fusion (dual-branch), Semantic-only (DeBERTa), "
            "and Stylometric-only (TF-IDF) representations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for dashboards, mail client add-ins, and MTA milters
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root endpoint to interactive OpenAPI Swagger documentation."""
        return RedirectResponse(url="/docs")

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        summary="Service Health & Model Status",
        tags=["Monitoring"],
    )
    async def health_check(request: Request) -> HealthResponse:
        """Check system readiness, active classifier head, and loaded device."""
        service: InferenceService = request.app.state.service
        return HealthResponse(
            status="healthy" if service.is_ready else "initializing",
            version="1.0.0",
            classifier=service.classifier_name,
            device=service.device,
            models_loaded={
                "stylometric_tfidf": service.stylometric_extractor is not None,
                "semantic_deberta": service.semantic_encoder is not None,
                "classifier_head": service.classifier_head is not None,
            },
        )

    @app.post(
        "/api/v1/predict",
        response_model=PredictResponse,
        summary="Predict from Structured Email Payload",
        tags=["Inference"],
    )
    async def predict_email(
        payload: PredictRequest,
        request: Request,
        mode: Optional[str] = Query(
            default=None,
            description="Optional representation mode: 'fusion' (dual-branch), 'semantic' (DeBERTa-only), or 'stylometric' (TF-IDF only). Overrides payload.mode if specified.",
        ),
    ) -> PredictResponse:
        """Analyze email subject and body for phishing and LLM provenance.

        Automatically applies HTML stripping, thread slicing, Unicode normalization,
        and PII/URL token masking before model inference in the requested representation mode.
        """
        service: InferenceService = request.app.state.service
        if not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference models are currently loading.",
            )

        if not payload.subject and not payload.body:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Either 'subject' or 'body' must be non-empty.",
            )

        effective_mode = mode or payload.mode or service.default_mode

        try:
            # Execute inference in worker threadpool to keep ASGI loop non-blocking
            response = await asyncio.to_thread(
                service.predict,
                payload.subject,
                payload.body,
                effective_mode,
            )
            return response
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(exc)}",
            )

    @app.post(
        "/api/v1/predict/raw",
        response_model=PredictResponse,
        summary="Predict from Raw RFC 5322 MIME Email (.eml)",
        tags=["Inference"],
    )
    async def predict_raw_mime_email(
        request: Request,
        file: Optional[UploadFile] = File(default=None),
        mode: Optional[str] = Query(
            default=None,
            description="Optional representation mode: 'fusion' (dual-branch), 'semantic' (DeBERTa-only), or 'stylometric' (TF-IDF only).",
        ),
    ) -> PredictResponse:
        """Analyze raw RFC 5322 MIME email payload (or uploaded .eml file).

        Parses MIME boundaries, extracts subject and plain/HTML body parts,
        cleans and masks content, and returns classification confidences in the requested mode.
        """
        service: InferenceService = request.app.state.service
        if not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference models are currently loading.",
            )

        # Read content from file upload or raw request body
        if file is not None:
            raw_bytes = await file.read()
        else:
            raw_bytes = await request.body()

        if not raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty payload. Please upload an .eml file or supply raw MIME bytes in the request body.",
            )

        try:
            subject, body = parse_rfc822_email(raw_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse RFC 5322 MIME message: {str(exc)}",
            )

        effective_mode = mode or service.default_mode

        try:
            response = await asyncio.to_thread(
                service.predict,
                subject,
                body,
                effective_mode,
            )
            return response
        except Exception as exc:
            logger.error("Raw inference failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(exc)}",
            )

    return app

