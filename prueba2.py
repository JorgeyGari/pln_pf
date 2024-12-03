import os
os.environ['OLLAMA_HOST'] = 'http://kumo01.tsc.uc3m.es:11434'

import ollama

llm_model = 'llama3.2'
response = ollama.chat(
	model=llm_model,
	messages=[
		{
			'role': 'user',
			'content': 'Describe me what topic modeling is as if I were a kid',
		},
	])
print(response['message']['content'])
