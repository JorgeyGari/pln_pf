import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import wikipediaapi
from sentence_transformers import SentenceTransformer

# Step 1: Initialize ChromaDB
chroma_client = chromadb.Client()  # Updated client initialization

# Step 2: Define Embedding Function
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Lightweight transformer model
)

# Create a collection in ChromaDB
wiki_collection = chroma_client.get_or_create_collection(
    name="wikipedia",
    embedding_function=embedding_function
)

# Step 3: Fetch Data from Wikipedia
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="WikipediaFetcherBot (your-email@example.com)"  # Specify user agent
)

def fetch_wikipedia_summary(title):
    page = wiki.page(title)
    if page.exists():
        return page.summary
    else:
        return None

# Example Titles to Fetch
titles = ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]

# Step 4: Prepare Data and Insert into ChromaDB
for title in titles:
    summary = fetch_wikipedia_summary(title)
    if summary:
        wiki_collection.add(
            documents=[summary],  # Content to be embedded
            metadatas=[{"title": title}],  # Metadata for each entry
            ids=[title]  # Unique ID for the entry
        )

print("Data added to ChromaDB successfully!")

