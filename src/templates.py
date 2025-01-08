# This file contains the templates for the task descriptions.
get_relevant_sections = """
    Identify the most relevant sections from a set of pages to answer the given question.

    The second input given is a list of tuples of pages and sections in the following format:
    [(name_of_page1, [section1, section2, ...]), (name_of_page2, [section1, ...]), ...]

    Inputs:
    1. **Question:** "{}"
    2. **Pages and Sections:** {}

    Task:
    Determine the pages most relevant to answer the question. Specify the sections that are most relevant to answer the question. Keep the number of sections to a minimum. You may repeat pages if necessary to retrieve multiple sections from the same page.
    """

questions = """Based on the following information:\n{context}. \n\nAnswer the following question or say if it's true or false: {question}. Answer in a concise manner, use a maximum of three sentences.\n\n"""

answer = "In light of the above reasoning, how would you answer the question '{}'?\nPlease respond in {}."
