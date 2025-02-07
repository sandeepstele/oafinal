import numpy as np
import faiss
import openai
import os

# Make sure your AI proxy is configured, if needed
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small", 
        input=[text]
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

# Load persistent documents from file
def load_persistent_docs(file_path="persistent_docs.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs

docs = load_persistent_docs()  # Adjust path if needed

# Compute embeddings for each document
embeddings = [get_embedding(doc) for doc in docs]
embeddings = np.vstack(embeddings)
embedding_dim = embeddings.shape[1]

# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save the index to a file
faiss.write_index(index, "persistent_index.index")
print("FAISS index created and saved as persistent_index.index")
