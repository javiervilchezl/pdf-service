from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app
import app.api
import app.core
import app.providers
import app.schemas
import app.services
from app.api.routes import get_pdf_service
from app.core.config import settings
from app.main import app as fastapi_app
from app.providers.base import LLMProvider
from app.providers.factory import get_provider
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider
from app.schemas.pdf import PDFAnalysisResponse
from app.services.pdf_processing import (
    InvalidPDFError,
    InvalidProviderResponseError,
    PDFProcessingService,
)


class DummyProvider(LLMProvider):
    async def generate(self, prompt: str, system_prompt: str) -> str:
        return await super().generate(prompt, system_prompt)


class StubPDFService:
    def __init__(self, extracted: str = "hello world") -> None:
        self.extracted = extracted

    def extract_text(self, payload: bytes) -> str:
        return self.extracted

    def clean_text(self, text: str) -> str:
        return text.upper()

    async def summarize_and_tag(self, text: str) -> PDFAnalysisResponse:
        return PDFAnalysisResponse(
            text=text,
            summary="summary",
            topics=["finance"],
        )


class StubInvalidPDFService(StubPDFService):
    def extract_text(self, payload: bytes) -> str:
        raise InvalidPDFError("Invalid or unreadable PDF file")


class StubInvalidProviderResponseService(StubPDFService):
    async def summarize_and_tag(self, text: str) -> PDFAnalysisResponse:
        raise InvalidProviderResponseError(
            "AI provider returned an invalid response for PDF analysis"
        )


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    fastapi_app.dependency_overrides.clear()
    yield
    fastapi_app.dependency_overrides.clear()


def test_package_exports_are_importable() -> None:
    assert app.__all__ == []
    assert app.api.__all__ == []
    assert app.core.__all__ == []
    assert app.providers.__all__ == []
    assert app.schemas.__all__ == []
    assert app.services.__all__ == []


@pytest.mark.asyncio
async def test_base_provider_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await DummyProvider().generate("prompt", "system")


def test_get_provider_returns_openai(monkeypatch) -> None:
    from app.providers import factory

    monkeypatch.setattr(factory.settings, "provider", "openai")
    monkeypatch.setattr(factory, "OpenAIProvider", lambda: "openai")
    assert get_provider() == "openai"


def test_get_provider_returns_groq(monkeypatch) -> None:
    from app.providers import factory

    monkeypatch.setattr(factory.settings, "provider", "groq")
    monkeypatch.setattr(factory, "GroqProvider", lambda: "groq")
    assert get_provider() == "groq"


@pytest.mark.asyncio
async def test_openai_provider_generate(monkeypatch) -> None:
    captured = {}

    class FakeResponses:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="ok")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(
        "app.providers.openai_provider.AsyncOpenAI",
        FakeClient,
    )
    provider = OpenAIProvider()
    result = await provider.generate("hello", "system")

    assert result == "ok"
    assert captured["input"][0]["content"] == "system"
    assert captured["input"][1]["content"] == "hello"


@pytest.mark.asyncio
async def test_groq_provider_generate(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="groq-ok")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_module = SimpleNamespace(AsyncGroq=FakeClient)
    monkeypatch.setattr(
        "app.providers.groq_provider.import_module",
        lambda _: fake_module,
    )
    provider = GroqProvider()
    result = await provider.generate("hello", "system")

    assert result == "groq-ok"
    assert captured["messages"][0]["content"] == "system"
    assert captured["messages"][1]["content"] == "hello"


def test_extract_text(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self.text

    class FakeDocument:
        def __init__(self) -> None:
            self.closed = False
            self.pages = [FakePage("one"), FakePage("two")]

        def __iter__(self):
            return iter(self.pages)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "app.services.pdf_processing.fitz.open",
        lambda **_: FakeDocument(),
    )
    service = PDFProcessingService(provider=SimpleNamespace())
    assert service.extract_text(b"pdf") == "one\ntwo"


def test_extract_text_invalid_pdf(monkeypatch) -> None:
    def fake_open(**kwargs):
        raise RuntimeError("broken-pdf")

    monkeypatch.setattr("app.services.pdf_processing.fitz.open", fake_open)
    service = PDFProcessingService(provider=SimpleNamespace())

    with pytest.raises(InvalidPDFError):
        service.extract_text(b"pdf")


def test_extract_text_page_read_error(monkeypatch) -> None:
    class BrokenPage:
        def get_text(self, mode: str) -> str:
            raise RuntimeError("page-error")

    class FakeDocument:
        def __iter__(self):
            return iter([BrokenPage()])

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "app.services.pdf_processing.fitz.open",
        lambda **_: FakeDocument(),
    )
    service = PDFProcessingService(provider=SimpleNamespace())

    with pytest.raises(InvalidPDFError):
        service.extract_text(b"pdf")


def test_clean_text() -> None:
    service = PDFProcessingService(provider=SimpleNamespace())
    assert service.clean_text(" a\n\tb  c ") == "a b c"


@pytest.mark.asyncio
async def test_summarize_and_tag() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            assert "Analyze the document text" in prompt
            assert "valid JSON" in system_prompt
            return '{"summary":"brief","topics":["report"]}'

    service = PDFProcessingService(provider=FakeProvider())
    result = await service.summarize_and_tag("text")

    assert result.summary == "brief"
    assert result.topics == ["report"]
    assert result.text == "text"


@pytest.mark.asyncio
async def test_summarize_and_tag_accepts_fenced_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '```json\n{"summary":"brief","topics":["report"]}\n```'

    service = PDFProcessingService(provider=FakeProvider())
    result = await service.summarize_and_tag("text")

    assert result.summary == "brief"
    assert result.topics == ["report"]


@pytest.mark.asyncio
async def test_summarize_and_tag_rejects_invalid_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return "not-json"

    service = PDFProcessingService(provider=FakeProvider())

    with pytest.raises(InvalidProviderResponseError):
        await service.summarize_and_tag("text")


@pytest.mark.asyncio
async def test_summarize_and_tag_accepts_embedded_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return 'analysis:{"summary":"brief","topics":["report"]}'

    service = PDFProcessingService(provider=FakeProvider())
    result = await service.summarize_and_tag("text")

    assert result.summary == "brief"
    assert result.topics == ["report"]


@pytest.mark.asyncio
async def test_summarize_and_tag_rejects_invalid_embedded_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return 'analysis:{"summary":bad-json}'

    service = PDFProcessingService(provider=FakeProvider())

    with pytest.raises(InvalidProviderResponseError):
        await service.summarize_and_tag("text")


def test_health_endpoint() -> None:
    client = TestClient(fastapi_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_pdf_rejects_invalid_file_type() -> None:
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )
    assert response.status_code == 400


def test_analyze_pdf_rejects_empty_extraction() -> None:
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubPDFService("")
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 422


def test_analyze_pdf_rejects_invalid_pdf() -> None:
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubInvalidPDFService()
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unreadable PDF file"


def test_analyze_pdf_invalid_provider_response() -> None:
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubInvalidProviderResponseService("hello")
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 502


def test_analyze_pdf_success() -> None:
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubPDFService("hello")
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "HELLO",
        "summary": "summary",
        "topics": ["finance"],
    }


def test_analyze_pdf_requires_internal_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "svc-key")
    monkeypatch.setattr(settings, "internal_api_key_header", "X-Service-API-Key")
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubPDFService("hello")
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 401


def test_analyze_pdf_accepts_internal_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "svc-key")
    monkeypatch.setattr(settings, "internal_api_key_header", "X-Service-API-Key")
    fastapi_app.dependency_overrides[get_pdf_service] = (
        lambda: StubPDFService("hello")
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/analyze-pdf",
        files={"file": ("doc.pdf", b"pdf", "application/pdf")},
        headers={"X-Service-API-Key": "svc-key"},
    )
    assert response.status_code == 200
