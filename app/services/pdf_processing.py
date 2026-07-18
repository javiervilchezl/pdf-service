import io
import json
import re

import fitz

from app.providers.base import LLMProvider
from app.schemas.pdf import PDFAnalysisResponse


class InvalidPDFError(ValueError):
    pass


class InvalidProviderResponseError(ValueError):
    pass


class PDFProcessingService:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def extract_text(self, payload: bytes) -> str:
        try:
            document = fitz.open(stream=io.BytesIO(payload), filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError("Invalid or unreadable PDF file") from exc

        try:
            pages = [page.get_text("text") for page in document]
        except Exception as exc:
            raise InvalidPDFError("Invalid or unreadable PDF file") from exc
        finally:
            document.close()

        return "\n".join(filter(None, pages)).strip()

    def clean_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text)
        return compact.strip()

    async def summarize_and_tag(self, text: str) -> PDFAnalysisResponse:
        prompt = (
            "Analyze the document text. Return valid JSON with this shape: "
            '{"summary":"...","topics":["..."]}.\n\n'
            "Keep topics short and business-friendly. Text: "
            f"{text[:12000]}"
        )
        content = await self.provider.generate(
            prompt=prompt,
            system_prompt=(
                "You are an expert document analyst. "
                "Respond only with valid JSON."
            ),
        )
        parsed = self._parse_provider_response(content)
        return PDFAnalysisResponse(
            text=text,
            summary=parsed.get("summary", ""),
            topics=parsed.get("topics", []),
        )

    def _parse_provider_response(self, content: str) -> dict:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        raise InvalidProviderResponseError(
            "AI provider returned an invalid response for PDF analysis"
        )
