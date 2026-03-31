from pydantic import BaseModel, Field


class PDFAnalysisResponse(BaseModel):
    text: str = Field(..., description="Extracted and cleaned document text")
    summary: str = Field(..., description="Document summary")
    topics: list[str] = Field(
        default_factory=list,
        description="Main document topics",
    )
