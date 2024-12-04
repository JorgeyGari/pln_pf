import requests
import json

def generate_response(model, system, prompt, url="http://kumo01:11434/api/generate"):
    data = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}, {response.text}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

response = generate_response("llama3.2", "You are a helpful AI Assistant", "What are you?")
print(response["response"])
