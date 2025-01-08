# This file contains the templates for the task descriptions.
get_relevant_sections = """Identify the most relevant sections from a set of pages to answer the given question.

The second input given is a list of tuples of pages and sections in the following format:
[(name_of_page1, [section1, section2, ...]), (name_of_page2, [section1, ...]), ...]

Inputs:
1. **Question:** "{question}"
2. **Pages and Sections:** {sections_and_page}

Task:
Determine the sections that are most relevant to answer the question and respond strictly in the following format:
[section1, section2, ...]

Examples:
example_user: What is the capital of France?
example_sections: [(Paris, [Introduction, Geography, History]), (France, [Geography, History, Economy])]
example_assistant: [Geography, History]

example_user: Who is the author of the book "1984"?
example_sections: [(1984 (book), [Plot summary, Characters, Themes]), (George Orwell, [Biography, Works, Influence])]
example_assistant: [Biography, Works]

example_user: What is the population of Tokyo?
example_sections: [(Tokyo, [Geography, History, Demographics]), (Japan, [Geography, History, Demographics])]
example_assistant: [Demographics]
"""