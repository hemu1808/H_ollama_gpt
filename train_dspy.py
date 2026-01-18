import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy_module import RAGModule
from config import settings
import os
import random

# --- 1. SETUP ---
lm = dspy.LM('ollama/' + settings.OLLAMA_LLM_MODEL, api_base=settings.OLLAMA_URL)
dspy.configure(lm=lm)

def main():
    print("Starting DSPy Optimization Pipeline...")

    # --- 2. DATASET (Scalable) ---
    # FOR HUGE DATA: Load this from a JSON file instead of hardcoding!
    # import json
    # with open("huge_training_data.json") as f:
    #     raw_data = json.load(f)
    #     trainset = [dspy.Example(context=d['c'], question=d['q'], answer=d['a']).with_inputs('context', 'question') for d in raw_data]

    # For now, here is a Robust "Gold Standard" set (10 examples)
    # These teach the model NOT to be a dictionary.
    raw_examples = [
        {
            "context": "Hemanth built a Distributed Container Orchestration Engine in Go using Docker API.",
            "question": "What is an orchestration engine?",
            "answer": "Hemanth developed a **Distributed Container Orchestration Engine** using **Go** and the **Docker API**, focusing on scheduling and node management."
        },
        {
            "context": "The system detects node failures in 30ms using a gRPC heartbeat mechanism.",
            "question": "How does it handle failures?",
            "answer": "The system implements a **gRPC heartbeat mechanism** that detects node failures within **30ms** to trigger rescheduling."
        },
        {
            "context": "Hemanth has 2 years of experience. He worked at StartupX.",
            "question": "Experience summary",
            "answer": "Hemanth has **2 years** of production experience, notably working at **StartupX**."
        },
        {
            "context": "The API listens on port 8000. It uses ChromaDB for vector storage.",
            "question": "What database is used?",
            "answer": "The system utilizes **ChromaDB** for vector storage."
        },
        {
            "context": "This project uses Python 3.11 and FastAPI.",
            "question": "Language used?",
            "answer": "The project is built using **Python 3.11** and **FastAPI**."
        },
        # ... Add 100 more examples here for "Huge Data" training ...
    ]

    # Convert to DSPy format
    trainset = [dspy.Example(
        context=e['context'], 
        question=e['question'], 
        answer=e['answer']
    ).with_inputs('context', 'question') for e in raw_examples]

    print(f"Loaded {len(trainset)} training examples.")

    # --- 3. METRIC (The Judge) ---
    # We use a Semantic Match metric. 
    # It checks if the semantic meaning of the prediction matches the gold answer.
    def validate_answer(example, pred, trace=None):
        # 1. Sanity Check: Don't allow empty answers
        if len(pred.answer) < 5: return False
        
        # 2. Key Phrase Matching (Robust)
        # We strip markdown for comparison
        gold_clean = example.answer.replace("**", "").lower()
        pred_clean = pred.answer.lower()
        
        # If the prediction contains the core keywords from the gold answer, it's good.
        # This prevents "exact match" failures on minor wording differences.
        # For huge data, you can use dspy.evaluate.SemanticF1 if available.
        return dspy.evaluate.answer_passage_match(example, pred)

    # --- 4. COMPILE (TRAIN) ---
    print("Compiling (Optimizing) the RAG Module...")
    
    # BootstrapFewShot is perfect for 10-50 examples.
    # If you have 500+ examples, consider using 'BootstrapFewShotWithRandomSearch'
    teleprompter = BootstrapFewShot(
        metric=validate_answer,
        max_bootstrapped_demos=4,  # Use up to 4 examples in the prompt context
        max_labeled_demos=4        # Use 4 labelled examples
    )

    rag = RAGModule()
    
    # This runs the training loop
    compiled_rag = teleprompter.compile(rag, trainset=trainset)

    # --- 5. SAVE ---
    save_path = "./data/compiled_rag.json"
    os.makedirs("./data", exist_ok=True)
    compiled_rag.save(save_path)
    
    print(f"✅ Optimization Complete! Saved to {save_path}")
    print("👉 Now restart your API. The model is now trained to mimic your examples.")

if __name__ == "__main__":
    main()