"""
from langchain_community.emmbedding.bedrock import BedrockEmbeddings

def get_embeddings_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name = "default", region_name = "us-east-1"
    )

from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
"""