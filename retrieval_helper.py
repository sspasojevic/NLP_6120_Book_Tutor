import torch
import numpy as np
import faiss

def get_text_embedding(model, tokenizer, text):
    """
    Generates a text embedding by encoding the input text using the provided model and tokenizer.

    Args:
        model: The pre-trained language model used to generate embeddings.
        tokenizer: The tokenizer corresponding to the model, used to preprocess the text.
        text (str): The input text to be embedded.

    Returns:
        np.ndarray: A 1-dimensional numpy array representing the text embedding.
    """
    # Tokenize the input text, ensuring it's truncated to a maximum length of 512 tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Pass the tokenized input through the model to obtain the outputs
    outputs = model(**inputs)
    
    # Compute the mean of the last hidden states to get a fixed-size embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

def retrieve_top_k_similar(query, embeddings, embedding_model, embedding_tokenizer, top_k=3, method="faiss"):
    """
    Retrieves the top K most similar embeddings to a given query using either cosine similarity or FAISS.

    Args:
        query (str): The query text for which to find similar embeddings.
        embeddings (np.ndarray): A 2D numpy array where each row is an embedding.
        embedding_model: The model used to generate embeddings for the query.
        embedding_tokenizer: The tokenizer corresponding to the embedding model.
        top_k (int, optional): The number of top similar embeddings to retrieve. Defaults to 3.
        method (str, optional): The similarity method to use ("cosine" or "faiss"). Defaults to "faiss".

    Returns:
        tuple: A tuple containing two numpy arrays:
            - Similarity scores or distances of the top K embeddings.
            - Indices of the top K similar embeddings in the original embeddings array.
    """
    # Generate embedding for the query text
    query_embedding = get_text_embedding(embedding_model, embedding_tokenizer, query)

    if method == "cosine":
        # Convert the query embedding to a PyTorch tensor for cosine similarity computation
        query_embedding = torch.tensor(query_embedding)
        
        # Calculate cosine similarity between the query embedding and all embeddings
        similarity = torch.nn.functional.cosine_similarity(embeddings, query_embedding, dim=-1)
        
        # Retrieve the top K similarities and their corresponding indices
        similarity_top_k = torch.topk(similarity, k=top_k)
        
        return np.array(similarity_top_k.values), np.array(similarity_top_k.indices)

    elif method == "faiss":
        # Determine the dimensionality of the embeddings
        d = embeddings.shape[1]
        
        # Initialize a FAISS index for L2 (Euclidean) distance
        index = faiss.IndexFlatL2(d)
        
        # Add all embeddings to the FAISS index
        index.add(embeddings)
        
        # Reshape the query embedding to match FAISS input requirements
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        
        # Perform the search to find the top K nearest neighbors
        D, I = index.search(query_embedding_reshaped, k=top_k)
        
        return D.reshape(-1), I.reshape(-1)

def get_items_for_prompt(query, df, indices):
    """
    Constructs a dictionary containing the query and the top 3 related items from a DataFrame.

    Args:
        query (str): The original query text.
        df (pd.DataFrame): The DataFrame containing the data with 'page_number' and 'text' columns.
        indices (array-like): Indices of the top related items in the DataFrame.

    Returns:
        dict: A dictionary with the query and details of the top 3 related items, including page numbers and context texts.
    """
    # Create a dictionary with the query and the corresponding page numbers and texts for the top 3 indices
    dict1 = {
        "query": query,
        "page1": df["page_number"][indices[0]],
        "page2": df["page_number"][indices[1]],
        "page3": df["page_number"][indices[2]],
        "context1": df["text"][indices[0]],
        "context2": df["text"][indices[1]],
        "context3": df["text"][indices[2]]
    }

    return dict1
