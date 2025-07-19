import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

### Embedder base class
from abc import ABC

class Embedder(ABC):
    def transform(self, contract: str) -> np.array:
        pass

### BAAI Embedder
class BAAIEmbedder(Embedder):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

    def transform(self, contract: str) -> np.array:
        inputs = self.tokenizer(contract, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state[:, 0].squeeze().numpy()

### Load data
kalshi_data = pd.read_csv("kalshi_markets.csv")
polymarket_data = pd.read_csv("polymarket_markets.csv")

# Take first 3000 from each
kalshi_titles = kalshi_data["Title"][:3000].tolist()
polymarket_titles = polymarket_data["Title"][:3000].tolist()

### Initialize embedder
model = BAAIEmbedder()

### Embed titles
print("Embedding Kalshi titles...")
kalshi_embeddings = [model.transform(title) for title in kalshi_titles]

print("Embedding Polymarket titles...")
polymarket_embeddings = [model.transform(title) for title in polymarket_titles]

### Compute cosine similarity matrix
print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(kalshi_embeddings, polymarket_embeddings)

### Find best matches
top_indices = np.argmax(similarity_matrix, axis=1)
top_scores = np.max(similarity_matrix, axis=1)

### Print top matches above threshold
threshold = 0.92
print("\nTop Matches:")
for i, (score, j) in enumerate(zip(top_scores, top_indices)):
    if score >= threshold:
        print(f"With score={score:.3f} match found between:")
        print(f"Kalshi:     {kalshi_titles[i]}")
        print(f"Polymarket: {polymarket_titles[j]}\n")
