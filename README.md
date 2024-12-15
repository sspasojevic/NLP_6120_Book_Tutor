# Current project implementation via Gradio
https://huggingface.co/spaces/utkarshgogna1/Gradio_APP
(uses shared resources and it takes time to load and process)

# Project Setup Instructions

This document outlines the steps required to set up the project environment, install dependencies, and configure the necessary API token.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- `pip` (Python package manager)

---

## Setup Instructions

To isolate the project dependencies, create a virtual environment.
You will need to create and activate your own virtual environment which you can recreate by installing requirements.txt using:

pip install -r requirements.txt

Additionally, you will need to make your own .env file and add the HF_API_TOKEN to it (Serverless Inference API - https://huggingface.co/docs/api-inference/en/index):

touch .env
HF_API_TOKEN={your_token_here}

## Finetuned GPT2

To run the fine-tuned GPT2 part of the notebook, you will need to move the model's folder into the project folder.
Link to the fine-tuned saved model: https://drive.google.com/drive/folders/1Jm6PYyaZbzb57KNH4l8dbl7hRQe_xYA8?usp=drive_link


