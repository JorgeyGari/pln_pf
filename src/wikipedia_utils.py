import wikipediaapi


wiki_wiki = wikipediaapi.Wikipedia('FactChecker (email@example.com)', 'en')

def extract_text_from_sections(pages_sections_dict):
        """
        Given the relevant sections and a list of pages/sections,
        extract the corresponding text from those sections using the Wikipedia API.
        """
        relevant_texts = {}
        
        for page in pages_sections_dict:
            text = wiki_wiki.page(page["page_title"]).section_by_title(page["section"])
            if text:
                relevant_texts[page["section"] + " (" + page["page_title"] + ")"] = text
        return relevant_texts
