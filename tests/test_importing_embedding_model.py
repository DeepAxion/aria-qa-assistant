"""Testing importing embedding model"""
from langchain_community.llms import Ollama
print("Ollama imported successfully!")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
print(model.encode("hello").shape) # should print (384,)
