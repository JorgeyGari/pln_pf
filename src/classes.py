from pydantic import BaseModel
from typing import List

# Pydantic data models
class PageSection(BaseModel):
    """Data model for a Wikipedia page section."""
    name: str
    sections: List[str]

class LLMOutput(BaseModel):
    data: List[PageSection]
    