import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
PHI_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"
QWEN_API_URL = 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct'

def query_hf_mistral(prompt, api_url=MISTRAL_API_URL):
    """
    Sends the prompt to the Hugging Face Inference API and returns the response.

    Args:
        prompt (str): The input prompt to send to the model.
        api_url (str): The Hugging Face model API endpoint.

    Returns:
        str: The generated response from the model.
    """
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    prompt = "<s>[INST]" + prompt + " [/INST] Model answer</s>"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.8, "return_full_text": False }} # can play with these variables here
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse and return the response
    response_data = response.json()
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")
    
def query_hf_phi(prompt, api_url=PHI_API_URL):
    """
    Sends the prompt to the Hugging Face model and returns the response.

    Args:
        prompt (str): The input prompt to send to the model.
        api_url (str): The Hugging Face model API endpoint.

    Returns:
        str: The generated response from the model.
    """
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    prompt = prompt + "\n\nOutput:"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2, "return_full_text": False }}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse and return the response
    response_data = response.json()
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")
    
def query_hf_qwen(prompt, api_url=QWEN_API_URL):
    """
    Sends the prompt to the Hugging Face model and returns the response.

    Args:
        prompt (str): The input prompt to send to the model.
        api_url (str): The Hugging Face model API endpoint.

    Returns:
        str: The generated response from the model.
    """
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    prompt = "<|im_start|>system\n" + prompt + "<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2, "return_full_text": False }}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse and return the response
    response_data = response.json()
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")