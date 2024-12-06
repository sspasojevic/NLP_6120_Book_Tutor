import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Define the API endpoints for different Hugging Face models
MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
PHI_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"
QWEN_API_URL = 'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct'

def query_hf_mistral(prompt, api_url=MISTRAL_API_URL):
    """
    Sends a prompt to the Mistral model via Hugging Face Inference API and returns the generated response.

    Args:
        prompt (str): The input text prompt to send to the Mistral model.
        api_url (str, optional): The Hugging Face API endpoint for the Mistral model. Defaults to MISTRAL_API_URL.

    Returns:
        str: The response generated by the Mistral model.

    Raises:
        Exception: If the API request fails or returns a non-200 status code.
    """
    
    # Set up the authorization header with the Hugging Face API token
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Format the prompt according to the Mistral model's expected input structure
    prompt = "<s>[INST]" + prompt + " [/INST] Model answer</s>"
    
    # Define the payload with input prompt and generation parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,    # Maximum number of tokens to generate
            "temperature": 0.8,       # Sampling temperature for diversity
            "return_full_text": False # Whether to return the full text including the prompt
        }
    }
    
    # Send a POST request to the Hugging Face Inference API
    response = requests.post(api_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse the JSON response
    response_data = response.json()
    
    # Handle different response formats
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")
    
def query_hf_phi(prompt, api_url=PHI_API_URL):
    """
    Sends a prompt to the Phi model via Hugging Face Inference API and returns the generated response.

    Args:
        prompt (str): The input text prompt to send to the Phi model.
        api_url (str, optional): The Hugging Face API endpoint for the Phi model. Defaults to PHI_API_URL.

    Returns:
        str: The response generated by the Phi model.

    Raises:
        Exception: If the API request fails or returns a non-200 status code.
    """
    
    # Set up the authorization header with the Hugging Face API token
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Format the prompt according to the Phi model's expected input structure
    prompt = prompt + "\n\nOutput:"
    
    # Define the payload with input prompt and generation parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,    # Maximum number of tokens to generate
            "temperature": 0.2,       # Sampling temperature for less randomness
            "return_full_text": False # Whether to return the full text including the prompt
        }
    }
    
    # Send a POST request to the Hugging Face Inference API
    response = requests.post(api_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse the JSON response
    response_data = response.json()
    
    # Handle different response formats
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")
    
def query_hf_qwen(prompt, api_url=QWEN_API_URL):
    """
    Sends a prompt to the Qwen model via Hugging Face Inference API and returns the generated response.

    Args:
        prompt (str): The input text prompt to send to the Qwen model.
        api_url (str, optional): The Hugging Face API endpoint for the Qwen model. Defaults to QWEN_API_URL.

    Returns:
        str: The response generated by the Qwen model.

    Raises:
        Exception: If the API request fails or returns a non-200 status code.
    """
    
    # Set up the authorization header with the Hugging Face API token
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Format the prompt according to the Qwen model's expected input structure
    prompt = (
        "<|im_start|>system\n" + prompt +
        "<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    )
    
    # Define the payload with input prompt and generation parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,    # Maximum number of tokens to generate
            "temperature": 0.2,       # Sampling temperature for less randomness
            "return_full_text": False # Whether to return the full text including the prompt
        }
    }
    
    # Send a POST request to the Hugging Face Inference API
    response = requests.post(api_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Parse the JSON response
    response_data = response.json()
    
    # Handle different response formats
    if isinstance(response_data, list):  # Some models return a list of completions
        return response_data[0].get("generated_text", "No response available.")
    else:
        return response_data.get("generated_text", "No response available.")
