import wikipediaapi
import wikipedia
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms.base import LLM
from typing import Optional
from http import HTTPStatus
from fuzzywuzzy import process
import requests
import json

# Custom LLM
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

def search_wikipedia(query, language='en', top_n=5):
    wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', language)
    search_results = wiki.search(query)

    if not search_results:
        return f"No results found for '{query}'."
    
    matches = process.extract(query, search_results, limit=top_n)
    results = []
    for title, score in matches:
        page = wiki.page(title)
        if page.exists():
            results.append({
                'title': title,
                'summary': page.summary[:200] + '...' if len(page.summary) > 200 else  page.summary,
                'score': score
            }) 

    return results

def search_wikipedia(query, top_n=5):
    try:
        search_results = wikipedia.search(query)

        if not search_results:
            return f"No results found for '{query}'."
        

        results = []

        matches = process.extract(query, search_results, limit=top_n)

        for title, score in matches:
            try:
                page = wikipedia.page(title)
                results.append({
                    'title': title,
                    'summary': page.summary[:200] + '...' if len(page.summary) > 200 else page.summary,
                    'score': score 
                })
            except wikipedia.DisambiguationError as e:
                results.append({
                    'title': title,
                    'summary': "Disambiguation page. Options: " + ", ".join(e.options[:5] ) + "...",
                    'score': score
                })
            except wikipedia.PageError:
                results.append({
                    'title': title,
                    'summary': "Page not found.",
                    'score': score
                } )
        return results
    except Exception as e:
        return f"An error occurred: {e}"

def populate_chromadb_with_wikipedia(vector_store, query, threshold=50, top_n=5):
    search_results = search_wikipedia(query, top_n=top_n)

    documents = []
    for result in search_results:
        if result["score"] >= threshold:
            documents.append(Document(page_content=result["summary"], metadata={"title": result["title"]}))
    
    if documents:
        vector_store.add_documents(documents)
        vector_store.persist()
        print(f"Added {len(documents)} documents to the vector store.")
    else:
        print("No documents added. No results met the threshold.")

# Setup ChromaDB with SentenceTransformers embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="db/chroma", embedding_function=embedding)

# Define the QA chain for combining documents
combine_documents_chain = load_qa_chain(llm=llm, chain_type="stuff")

# Define the retreival-augmented QA system
retrieval_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Use the following context to answer the question:
        
        Context: {context}
        Question: {question}
        
        Answer in a concise and clear manner.
        """
)

retrieval_chain = RetrievalQA(
    retriever=vector_store.as_retriever(),
    combine_documents_chain=combine_documents_chain,
    return_source_documents=True,
)

# Example query
query = "Roman Empire collapse relevance"
response = retrieval_chain({"query": query})

# Display the response
print("Answer:", response["result"])
print(f"Source Documents ({len(response['source_documents'])}):")
for doc in response["source_documents"]:
    print(f"- {doc.page_content}")
