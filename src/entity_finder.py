from pydantic import BaseModel
import spacy
import wikipedia
from typing import List

# Initialize the NLP tool (spaCy)
nlp = spacy.load("en_core_web_sm")  # You can switch to a different model if needed


# Step 1: Define the Pydantic data models
class QueryResponse(BaseModel):
    titles: List[str]


# Step 2: Function to extract entities from the query using NLP
def extract_entities(query: str) -> List[str]:
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    return entities


# Step 3: Function to search Wikipedia for entities and return relevant titles
def search_wikipedia_for_entities(entities: List[str]) -> List[str]:
    titles = []

    for entity in entities:
        search_results = wikipedia.search(
            entity, results=3
        )  # Limiting to top 3 results
        titles.extend(search_results)  # Add titles to the list

    return titles


# Step 4: Chain function
def process_query(query: str) -> QueryResponse:
    # Extract entities
    entities = extract_entities(query)

    # If no entities found, return an empty response
    if not entities:
        return QueryResponse(titles=[])

    # Search Wikipedia for entities
    titles = search_wikipedia_for_entities(entities)

    # Return the response with titles
    return QueryResponse(titles=titles)


def list_entities(query: str) -> List[str]:
    list_entities = []
    response = process_query(query)
    for title in response.titles:
        list_entities.append(title)
    return list_entities


if __name__ == "__main__":
    query = "Mercury is the smallest planet in the Solar System."
    response = process_query(query)
    print(response)
