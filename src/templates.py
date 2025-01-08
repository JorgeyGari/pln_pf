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

confidence = """This is the question asked: {}\n
This is your answer: {}\n
And this is the information you used: {}\n
Based on this, how confident are you in your answer? You can base your response on the relevance of the information to the question, the clarity of the answer, and the overall quality of the response.
Provide a confidence score between 0 and 100, where 0 is not confident at all and 100 is very confident. Additionally, provide a brief reasoning for your confidence score.
Please respond in {}"""

summarize = """For each of the following sections: \n {context}, create a summary of the introduction of the page and each section, 
establishing at the beggining the name of the section and then a brief summary of the content.

Focus mainly on the most relevant information related to the question {question}. Do not mention subsections that are not related to the question. Please respond in {language}."""
