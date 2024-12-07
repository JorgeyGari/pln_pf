import ollama
import wikipediaapi
import wikipedia as wp
import re
import dspy
from IPython.display import display
import os

class FactsGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_facts = dspy.Predict(GenerateFacts)

    def forward(self, passage):
        return self.generate_facts(passage=passage).facts

class GenerateFacts(dspy.Signature):
    """
    Extract self-contained and fully contextualized facts from the given passage.
    """
    passage = dspy.InputField(
        desc="The passage may contain one or several claims"
    )
    facts = dspy.OutputField(
        desc="List of self-contained and fully contextualized claims in the form 'subject + verb + object' without using pronouns or vague references", 
        prefix="Facts:"
    )

def fetch_wikipedia_content(query, lang="en", top_n_results=1):
    """Fetch content from Wikipedia using a fuzzy search or entity-based queries."""
    user_agent = "YourAppName/1.0 (your_email@example.com)"
    wiki = wikipediaapi.Wikipedia(lang, headers={"User-Agent": user_agent})

    search_results = wp.search(query, results=top_n_results)

    if not search_results:
        print(f"No Wikipedia results found for query: {query}")
        return None, None

    for result in search_results:
        page = wiki.page(result)
        if page.exists():
            print(f"Fetched Wikipedia Page: {page.title}")
            print(f"Page URL: {page.fullurl}")
            return page.text, page.fullurl

    return None, None

def extract_entities(fact):
    """Extract key entities (e.g., names, places) from a fact using regex heuristics."""
    entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', fact)
    print(f"Extracted Entities: {entities}")
    return entities

def verify_fact(fact, facts_extractor, model="llama3.2"):
    """Verify a fact using enhanced Wikipedia content search."""
    print(f"\nVerifying Fact: {fact}")

    entities = extract_entities(fact)
    search_query = " ".join(entities) if entities else fact[:100]

    research, page_url = fetch_wikipedia_content(search_query)
    if research:
        print(f"\nWikipedia Content Extracted:\n{research[:500]}...")
    else:
        print("\nNo relevant information found on Wikipedia.")

    research_summary = research[:1000] if research else "No relevant information found on Wikipedia."

    print(f"\nResearch Summary:\n{research_summary}\n")

    prompt = f"""
    Verify the following fact based on trusted sources:
    Fact: "{fact}"
    Research: "{research_summary}"
    Response: Is this fact true, false, or unknown? Provide reasoning.
    """
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {'role': 'user',
             'content': f"prompt"},
        ]
    )

    print("\n--- Debugging End ---\n")
    print(f"Model Response:\n{response['message']['content']}\n")
    return {
        "fact": fact,
        "verification": response['message']['content'],
        "wikipedia_url": page_url
    }

def main():
    facts_extractor = FactsGeneratorModule()
    facts = [
        "Albert Einstein was born in Ulm, Germany, in 1879.",
        "I think horses have not one, not two, but three legs."
    ]

    results = []
    for fact in facts:
        result = verify_fact(fact, facts_extractor)
        results.append(result)

    print("\nResults:\n")
    for result in results:
        print(f"- Fact: {result['fact']}\n  Verification: {result['verification']}\n  Wikipedia URL: {result['wikipedia_url']}\n")

if __name__ == "__main__":
    os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'
    main()
