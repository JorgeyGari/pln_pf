from langchain.llms.base import LLM
from typing import Optional
from http import HTTPStatus

API_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
VERBOSE = False
ALLOW_WARNINGS = False


class CustomLLM(LLM):
    """
    Custom LLM class for the FactChecker task.
    Uses the Llama model deployed and accessible via the API URL.
    """

    model: str
    system: str
    api_url: str = API_URL

    def _call(
        self, prompt: str, stop: Optional[list] = None, format: Optional[dict] = None
    ) -> str:
        import requests
        import json

        data = {
            "model": self.model,
            "system": self.system,
            "prompt": prompt,
            "stream": False,
        }
        if stop:
            data["stop"] = stop
        if format:
            data["format"] = format
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self.api_url, headers=headers, data=json.dumps(data)
            )
            if response.status_code == HTTPStatus.OK:
                result = response.json()
                return result.get("response", "")
            else:
                raise ValueError(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            raise ValueError(f"An error occurred: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self):
        return {"model": self.model, "system": self.system}


llm = CustomLLM(
    model=MODEL,
    system="You are a helpful AI Assistant",
)

section_format = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_title": {"type": "string"},
                    "section": {
                        "type": "string",
                    },
                },
                "required": ["page_title", "section"],
            },
        }
    },
}

confidence_format = {
    "type": "object",
    "properties": {
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": ["confidence", "reasoning"],
}
