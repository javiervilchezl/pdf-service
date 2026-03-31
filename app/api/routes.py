from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.core.config import settings
from app.providers.factory import get_provider
from app.schemas.pdf import PDFAnalysisResponse
from app.services.pdf_processing import (
    InvalidPDFError,
    InvalidProviderResponseError,
    PDFProcessingService,
)

router = APIRouter()


def get_pdf_service() -> PDFProcessingService:
    return PDFProcessingService(provider=get_provider())


def verify_internal_api_key(
    api_key: str | None = Header(
        default=None,
        alias=settings.internal_api_key_header,
    ),
) -> None:
    if not settings.internal_api_key:
        return
    if api_key == settings.internal_api_key:
        return
    raise HTTPException(status_code=401, detail="Missing or invalid internal API key")


@router.post("/analyze-pdf", response_model=PDFAnalysisResponse)
async def analyze_pdf(
    file: UploadFile = File(...),
    _: None = Depends(verify_internal_api_key),
    service: PDFProcessingService = Depends(get_pdf_service),
) -> PDFAnalysisResponse:
    supported_types = {"application/pdf", "application/octet-stream"}
    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    payload = await file.read()
    try:
        extracted = service.extract_text(payload)
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not extracted:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in PDF",
        )

    cleaned = service.clean_text(extracted)
    try:
        return await service.summarize_and_tag(cleaned)
    except InvalidProviderResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
