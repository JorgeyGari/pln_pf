from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional
from http import HTTPStatus

class CustomLLM(LLM):
    model: str
    system: str
    api_url: str = "http://kumo01:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        """Make a call to the LLM's API and return the output."""
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
        """Return a string identifier for the LLM type."""
        return "custom_llm"
        
    @property
    def _identifying_params(self):
        """Return model-specific identifying parameters."""
        return {"model": self.model, "system": self.system}

llm = CustomLLM(
    model="llama3.2",
    system="You are a helpful AI Assistant",
)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}" 
)

chain = LLMChain(llm=llm, prompt=prompt_template)

question = "What are you?"
response = chain.run(question=question)
print(response)