import types
import torch
import pandas as pd
from retrieval_helper import *
from preprocessing_helper import *
from model_helper import *
from transformers import AutoTokenizer, AutoModel

def load_template(file_path):
    """Loads the template from a text file."""
    with open(file_path, 'r') as file:
        return file.read()
    
TEMPLATE = load_template('default_template.txt')

def ensure_list(item, name):
    if isinstance(item, str):
        item = [item]  # Wrap single string in a list
    elif not isinstance(item, list):
        raise ValueError(name + " must be passed as a string or a list of strings.")
    
    return item

def process_query(query, df, embedding_name, embedding_model, embedding_tokenizer, top_k=5, similarity_method="cosine"):
    embeddings = torch.tensor(np.array(df[embedding_name].tolist()), dtype=torch.float32)
    values, indices = retrieve_top_k_similar(query, embeddings, embedding_model, embedding_tokenizer, top_k=top_k, method=similarity_method)
    return indices, values

def generate_prompts(query, indices, df, templates):
    dict1 = get_items_for_prompt(query, df, indices)
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
    results = []
    for i, prompt in enumerate(prompts):
        for func in model_functions:
            response = func(prompt)
            results.append({
                "prompt_number": i + 1,
                "model_function": func.__name__,
                "response": response
            })
    return results

def compare_pipeline_configurations(queries, pdf_path, chunking_method=["sentence"], 
                                    embedding_tokenizers=["all-mpnet-base-v2"], 
                                    similarity_method="cosine", templates=[TEMPLATE], model_functions=[query_hf_mistral]):
    
    # --- Ensuring Proper Inputs ---
    
    queries = ensure_list(queries, "Queries")
    chunking_method = ensure_list(chunking_method, "Chunking method")
    embedding_tokenizers = ensure_list(embedding_tokenizers, "Embedding tokenizer")
    templates = ensure_list(templates, "Template")
    
    if isinstance(model_functions, types.FunctionType):
        model_functions = [model_functions]
    elif not all(isinstance(fn, types.FunctionType) for fn in model_functions):
        raise ValueError("model_functions must be a function or a list of functions.")
    
    # --- Creating dataframes and tokenizers ---
    
    dataframes = {}
    embedding_tokenizers_dict = {}

    # Create a df for every method of chunking
    for method in chunking_method:
        df=create_df_from_pdf(pdf_path, method=method)
        dataframes[method] = df
       
       
    # For every embedding tokenizer passed, embed text in each df and store model and tokenizer in a dict for later use
    for name in embedding_tokenizers:
        
        embedding_tokenizer_name = f"sentence-transformers/{name}"
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_name)
        embedding_model = AutoModel.from_pretrained(embedding_tokenizer_name)
        embedding_tokenizers_dict[name] = [embedding_tokenizer, embedding_model]
            
        for method, df in dataframes.items():

            embeddings = []

            for _, item in df.iterrows():
                embeddings.append(get_text_embedding(embedding_model, embedding_tokenizer, item["text"]))

            df[name] = embeddings
            
    # --- Create configurations ---
            
    pipeline_configurations = [
        (query, method, name, df)
        for query in queries
        for method, df in dataframes.items()
        for name in embedding_tokenizers
    ]     
    
    table_data = []
    
    # --- Get responses for all configurations and append to table ---
        
    for query, method, name, df in pipeline_configurations:
    # Process query and retrieve top-k results
        indices, _ = process_query(query, df, name, embedding_tokenizers_dict[name][1], embedding_tokenizers_dict[name][0])

        # Generate prompts
        prompts = generate_prompts(query, indices, df, templates)

        # Evaluate prompts
        responses = evaluate_prompts(prompts, model_functions)

        # Log or print results
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

    # Create a DataFrame
    results_df = pd.DataFrame(table_data)

    return results_df
