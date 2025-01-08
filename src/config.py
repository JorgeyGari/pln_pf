from langchain.llms.base import LLM
from typing import Optional
from http import HTTPStatus

class CustomLLM(LLM):
    model: str
    system: str
    api_url: str = "http://localhost:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[list] = None, format: Optional[dict] = None) -> str:
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
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
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

# Example usage with structured output
llm = CustomLLM(
    model="llama3.2:latest",
    system="You are a helpful AI Assistant",
)

# Define the structure (as per the cURL example)
format = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string"
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "title",
        "sections", 
    ]
}

