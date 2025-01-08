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

assumptions = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement based on the following documents:
    {context}\n\n"""

fact_checker = """Here is the bullet point list of assertions: \n
    {assertions}\n\n
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""

questions = """Given the following question: {question}\n Answer it based on the following information:\n{context}"""

answer = "In light of the above facts, how would you answer the question '{}'?\nPlease respond in {}."
