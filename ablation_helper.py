import types
import torch
import pandas as pd
from retrieval_helper import *
from preprocessing_helper import *
from model_helper import *
from transformers import AutoTokenizer, AutoModel

def load_template(file_path):
    """
    Loads a template from a specified text file.

    Args:
        file_path (str): The path to the template text file.

    Returns:
        str: The content of the template file as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()
        
# Load the default template from 'default_template.txt'
TEMPLATE = load_template('default_template.txt')

def ensure_list(item, name):
    """
    Ensures that the input item is a list. If it's a string, it wraps it in a list.
    Raises an error if the item is neither a string nor a list.

    Args:
        item (str or list): The input item to ensure as a list.
        name (str): The name of the variable for error messages.

    Returns:
        list: The input item as a list.
    
    Raises:
        ValueError: If the item is neither a string nor a list.
    """
    if isinstance(item, str):
        item = [item]  # Wrap single string in a list
    elif not isinstance(item, list):
        raise ValueError(name + " must be passed as a string or a list of strings.")
    
    return item

def process_query(query, df, embedding_name, embedding_model, embedding_tokenizer, top_k=5, similarity_method="cosine"):
    """
    Processes a query by retrieving the top K similar embeddings from a DataFrame.

    Args:
        query (str): The input query string.
        df (pd.DataFrame): The DataFrame containing embeddings and associated data.
        embedding_name (str): The column name in the DataFrame where embeddings are stored.
        embedding_model: The model used to generate embeddings for the query.
        embedding_tokenizer: The tokenizer corresponding to the embedding model.
        top_k (int, optional): The number of top similar results to retrieve. Defaults to 5.
        similarity_method (str, optional): The similarity method to use ("cosine" or "faiss"). Defaults to "cosine".

    Returns:
        tuple: A tuple containing:
            - indices (np.ndarray): The indices of the top K similar embeddings.
            - values (np.ndarray): The similarity scores or distances of the top K embeddings.
    """
    # Convert the list of embeddings in the DataFrame to a PyTorch tensor
    embeddings = torch.tensor(np.array(df[embedding_name].tolist()), dtype=torch.float32)
    
    # Retrieve the top K similar embeddings using the specified similarity method
    values, indices = retrieve_top_k_similar(query, embeddings, embedding_model, embedding_tokenizer, top_k=top_k, method=similarity_method)
    return indices, values

def generate_prompts(query, indices, df, templates):
    """
    Generates prompts by formatting templates with retrieved data based on query results.

    Args:
        query (str): The original query string.
        indices (np.ndarray): The indices of the top relevant entries in the DataFrame.
        df (pd.DataFrame): The DataFrame containing the data.
        templates (list of str): A list of template strings to format with data.

    Returns:
        list of str: A list of formatted prompt strings.
    """
    # Retrieve the relevant data for the prompt using the provided indices
    dict1 = get_items_for_prompt(query, df, indices)
    
    # Format each template with the retrieved data
    prompts = [
        template.format(
            page1=dict1["page1"], page2=dict1["page2"], page3=dict1["page3"],
            context1=dict1["context1"], context2=dict1["context2"], context3=dict1["context3"],
            query=dict1["query"]
        )
        for template in templates
    ]
    return prompts

def evaluate_prompts(prompts, model_functions):
    """
    Evaluates a list of prompts using specified model functions and collects the responses.

    Args:
        prompts (list of str): The list of prompt strings to evaluate.
        model_functions (list of callable): A list of functions that take a prompt string and return a response.

    Returns:
        list of dict: A list of dictionaries containing prompt number, model function name, and response.
    """
    results = []
    for i, prompt in enumerate(prompts):
        for func in model_functions:
            # Call the model function with the current prompt to get a response
            response = func(prompt)
            # Append the response details to the results list
            results.append({
                "prompt_number": i + 1,
                "model_function": func.__name__,
                "response": response
            })
    return results

def compare_pipeline_configurations(queries, pdf_path, chunking_method=["sentence"], 
                                    embedding_tokenizers=["all-mpnet-base-v2"], 
                                    similarity_method="cosine", templates=[TEMPLATE], model_functions=[query_hf_mistral]):
    """
    Compares different pipeline configurations by processing queries against a PDF using various chunking methods,
    embedding tokenizers, similarity methods, and templates, and evaluates responses from different models.

    Args:
        queries (str or list of str): The query or list of queries to process.
        pdf_path (str): The file path to the PDF document to be processed.
        chunking_method (str or list of str, optional): The method(s) to split PDF text into chunks ("sentence", "paragraph", etc.). Defaults to ["sentence"].
        embedding_tokenizers (str or list of str, optional): The tokenizer(s) to use for generating embeddings. Defaults to ["all-mpnet-base-v2"].
        similarity_method (str, optional): The similarity method to use ("cosine" or "faiss"). Defaults to "cosine".
        templates (str or list of str, optional): The template(s) to use for generating prompts. Defaults to [TEMPLATE].
        model_functions (callable or list of callable, optional): The model function(s) to generate responses. Defaults to [query_hf_mistral].

    Returns:
        pd.DataFrame: A DataFrame containing the results of all configurations, including responses from models.
    """
    
    # --- Ensuring Proper Inputs ---
    
    # Ensure that queries, chunking_method, embedding_tokenizers, and templates are lists
    queries = ensure_list(queries, "Queries")
    chunking_method = ensure_list(chunking_method, "Chunking method")
    embedding_tokenizers = ensure_list(embedding_tokenizers, "Embedding tokenizer")
    templates = ensure_list(templates, "Template")
    
    # Ensure that model_functions is a list of callable functions
    if isinstance(model_functions, types.FunctionType):
        model_functions = [model_functions]
    elif not all(isinstance(fn, types.FunctionType) for fn in model_functions):
        raise ValueError("model_functions must be a function or a list of functions.")
    
    # --- Creating DataFrames and Tokenizers ---
    
    dataframes = {}
    embedding_tokenizers_dict = {}

    # Create a DataFrame for each specified chunking method
    for method in chunking_method:
        df = create_df_from_pdf(pdf_path, method=method)
        dataframes[method] = df
       
    # For each embedding tokenizer, generate embeddings for each DataFrame and store the tokenizer and model
    for name in embedding_tokenizers:
        
        # Define the tokenizer name based on the embedding tokenizer
        embedding_tokenizer_name = f"sentence-transformers/{name}"
        
        # Load the tokenizer and model from the Hugging Face library
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_name)
        embedding_model = AutoModel.from_pretrained(embedding_tokenizer_name)
        
        # Store the tokenizer and model in a dictionary for later use
        embedding_tokenizers_dict[name] = [embedding_tokenizer, embedding_model]
            
        # Generate embeddings for each DataFrame using the current tokenizer and model
        for method, df in dataframes.items():

            embeddings = []

            for _, item in df.iterrows():
                # Generate embedding for the text in the current row
                embeddings.append(get_text_embedding(embedding_model, embedding_tokenizer, item["text"]))

            # Add the generated embeddings as a new column in the DataFrame
            df[name] = embeddings
            
    # --- Create Configurations ---
            
    # Create all possible combinations of query, chunking method, and embedding tokenizer
    pipeline_configurations = [
        (query, method, name, df)
        for query in queries
        for method, df in dataframes.items()
        for name in embedding_tokenizers
    ]     
    
    table_data = []
    
    # --- Get Responses for All Configurations and Append to Table ---
        
    for query, method, name, df in pipeline_configurations:
        # Process the query and retrieve the top K results based on embeddings and similarity method
        indices, _ = process_query(query, df, name, embedding_tokenizers_dict[name][1], embedding_tokenizers_dict[name][0])

        # Generate prompts using the retrieved indices and templates
        prompts = generate_prompts(query, indices, df, templates)

        # Evaluate the prompts using the specified model functions to get responses
        responses = evaluate_prompts(prompts, model_functions)

        # Log or store the results for each response
        for i, response in enumerate(responses):
            table_data.append({
                "Query": query,
                "Chunking Method": method,
                "Embedding": name,
                "Prompt #": response["prompt_number"],
                "Model Function": response["model_function"],
                "Response": response["response"],
                "Indices": indices
            })

    # Create a DataFrame from the collected table data
    results_df = pd.DataFrame(table_data)

    return results_df
