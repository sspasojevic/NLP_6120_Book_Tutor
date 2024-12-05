import torch
import numpy as np
import faiss

def get_text_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

def retrieve_top_k_similar(query, embeddings, embedding_model, embedding_tokenizer, top_k=3, method="faiss"):

  query_embedding = get_text_embedding(embedding_model, embedding_tokenizer, query)

  if method == "cosine":
    query_embedding = torch.tensor(query_embedding)

    similarity = torch.nn.functional.cosine_similarity(embeddings, query_embedding, dim=-1)
    similarity_top_k = torch.topk(similarity, k=top_k)

    return np.array(similarity_top_k.values), np.array(similarity_top_k.indices)

  # FAISS uses Euclidean distance
  elif method == "faiss":
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    query_embedding_reshaped = query_embedding.reshape(1, -1)

    D, I = index.search(query_embedding_reshaped, k=top_k)

    return D.reshape(-1), I.reshape(-1)

def get_items_for_prompt(query, df, indices):
  dict1 = {"query": query,
          "page1": df["page_number"][indices[0]],
          "page2": df["page_number"][indices[1]],
          "page3": df["page_number"][indices[2]],
          "context1": df["text"][indices[0]],
          "context2": df["text"][indices[1]],
          "context3": df["text"][indices[2]]
          }

  return dict1

