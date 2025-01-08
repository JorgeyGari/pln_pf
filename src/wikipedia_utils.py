import wikipediaapi

from entity_finder import list_entities


wiki_wiki = wikipediaapi.Wikipedia("FactChecker (email@example.com)", "en")


def extraer_titulos_sections(page):
    array_titulos = []
    sections = page.sections
    for s in sections:
        array_titulos.append(s.title)
    return array_titulos


def buscar_en_wikipedia(statement):
    titulos = list_entities(statement)

    if not titulos:
        return None, []

    paginas_contenido = []

    for titulo in titulos:
        page = wiki_wiki.page(titulo)
        if page.exists():
            pagina = page
            paginas_contenido.append((titulo, pagina))

    paginas_recuperadas = [pagina[1] for pagina in paginas_contenido]
    fuentes = [pagina[0] for pagina in paginas_contenido]

    return paginas_recuperadas, fuentes


def extract_text_from_sections(pages_sections_dict):
    """
    Given the relevant sections and a list of pages/sections,
    extract the corresponding text from those sections using the Wikipedia API.
    """
    relevant_texts = {}

    for page in pages_sections_dict:
        wiki_page = wiki_wiki.page(page["page_title"])
        if wiki_page.exists():
            text = wiki_page.section_by_title(page["section"])
            if text:
                relevant_texts[page["section"] + " (" + page["page_title"] + ")"] = text
    return relevant_texts
