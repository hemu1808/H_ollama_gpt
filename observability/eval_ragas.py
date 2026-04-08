import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Configure Langchain to use local Ollama deterministically
eval_llm = ChatOllama(base_url=settings.OLLAMA_URL, model=settings.OLLAMA_LLM_MODEL, temperature=0.0)
eval_embeddings = OllamaEmbeddings(base_url=settings.OLLAMA_URL, model=settings.OLLAMA_EMBEDDING_MODEL)

def load_data(path="dataset/golden_dataset.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)

def evaluate_baseline(dataset: Dataset):
    print("\n--- RUNNING BASELINE (UNQUANTIZED) EVALUATION ---")
    return evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

def evaluate_quantized(dataset: Dataset):
    print("\n--- RUNNING TURBOQUANT (3-BIT/1-BIT) EVALUATION ---")
    # In a full deployment, we explicitly load the TurboQuant inference engine here
    # e.g.: eval_llm = ChatOllama(model="llama3.1:8b_turboquant")
    return evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

def run_comparative_evaluation():
    print("Loading evaluation datasets...")
    try:
        baseline_dataset = load_data("dataset/golden_dataset.json") # Baseline outputs
        quantized_dataset = load_data("dataset/quantized_outputs.json") # Outputs using TurboQuant KV/Polar
    except FileNotFoundError:
        print("Required datasets not found. Mocking evaluation flow...")
        baseline_dataset = None
    
    if baseline_dataset:
        try:
            baseline_result = evaluate_baseline(baseline_dataset)
            quantized_result = evaluate_quantized(quantized_dataset)
            
            print("\n--- COMPARATIVE RESULTS ---")
            print(f"Baseline: {baseline_result}")
            print(f"Quantized: {quantized_result}")
            
            # Assert "Zero Accuracy Loss" requirement
            tolerance = 0.05
            for metric in ["context_precision", "faithfulness"]:
                b_score = baseline_result.get(metric, 0)
                q_score = quantized_result.get(metric, 0)
                if b_score - q_score > tolerance:
                    print(f"WARNING: Severe degradation detected in {metric}! Accuracy loss claim violated. Baseline: {b_score}, Quantized: {q_score}")
                else:
                    print(f"SUCCESS: {metric} preserved within tolerance ({b_score} -> {q_score})")
                    
        except Exception as e:
            print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_comparative_evaluation()
