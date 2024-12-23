from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
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

# Step 1: Question chain
question = "The Roman Empire collapsed in the 5th Century."
template_q = """{question}\n\n"""
prompt_template_q = PromptTemplate(input_variables=["question"], template=template_q)
question_chain = LLMChain(llm=llm, prompt=prompt_template_q)

# Step 2: Assumptions chain
template_assumptions = """Here is a statement:
{statement}
Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
prompt_template_assumptions = PromptTemplate(input_variables=["statement"], template=template_assumptions)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template_assumptions)

# Step 3: Assumptions chain
template_fact_checker = """Here is the bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template_fact_checker = PromptTemplate(input_variables=["assertions"], template=template_fact_checker)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template_fact_checker)

# Step 4: Answer chain based on verified facts
template_answer = """In light of the above facts, how would you answer the question '{}'""".format(question)
template_answer = """{facts}\n""" + template_answer
prompt_template_answer = PromptTemplate(input_variables=["facts"], template=template_answer)
answer_chain = LLMChain(llm=llm, prompt=prompt_template_answer)

# Step 5: Combine all the chains into a sequential workflow
overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True 
)

# Execute the entire workflow
print(f"Question: {question}\n")
final_response = overall_chain.run(question)
print(final_response)
