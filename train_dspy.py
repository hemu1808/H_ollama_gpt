import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
import logging

# Import your existing module (make sure to delete the old JSON first so it loads raw!)
from dspy_module import RAGModule 

# Set up DSPy LLM connection (use the same one from your dspy_module)
from config import settings
llm = dspy.LM(
    model='ollama/' + settings.OLLAMA_LLM_MODEL, 
    api_base=settings.OLLAMA_URL,
    api_key="ollama",
    temperature=0.1
)
dspy.configure(lm=llm)

class QuantizedRAGSignature(dspy.Signature):
    """Signature optimized for 3-bit quantized context"""
    context = dspy.InputField(desc="Compressed polar-quantized document chunks")
    question = dspy.InputField()
    answer = dspy.OutputField()

# --- 1. THE BALANCED DATASET ---
# We define specific inputs and the exact output we expect the LLM to generate.
trainset = [
    # Category 1: Portfolio & Personal Context
    dspy.Example(
        context="Hemanth built HGPT, a full-stack RAG application using Python and FastAPI.",
        context_chunk="Hemanth built HGPT, a full-stack RAG application using Python and FastAPI.",
        is_relevant="Yes",
        question="What did Hemanth build?",
        answer="Hemanth built HGPT, a RAG application using Python and FastAPI."
    ).with_inputs("context", "question", "context_chunk"),
    
    dspy.Example(
        context="Hemanth is actively looking for Software Engineer roles as of March 2026.",
        context_chunk="Hemanth is actively looking for Software Engineer roles as of March 2026.",
        is_relevant="Yes",
        question="Is the author looking for a job?",
        answer="Yes, Hemanth is actively looking for Software Engineer roles."
    ).with_inputs("context", "question", "context_chunk"),

    # Category 2: General Technical Knowledge (Fixes the KNN issue)
    dspy.Example(
        context="The K-Nearest Neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems.",
        context_chunk="The K-Nearest Neighbors (KNN) algorithm is a simple, supervised machine learning algorithm.",
        is_relevant="Yes",
        question="What is knn?",
        answer="KNN is a supervised machine learning algorithm used for classification and regression tasks."
    ).with_inputs("context", "question", "context_chunk"),

    dspy.Example(
        context="ChromaDB is an open-source vector database designed for AI applications.",
        context_chunk="ChromaDB is an open-source vector database designed for AI applications.",
        is_relevant="Yes",
        question="Explain ChromaDB.",
        answer="ChromaDB is an open-source vector database used for AI applications."
    ).with_inputs("context", "question", "context_chunk"),

    # Category 3: The "I Don't Know" Fallbacks (Prevents hallucinations)
    dspy.Example(
        context="The project uses Python 3.11 and Docker for containerization.",
        context_chunk="The project uses Python 3.11 and Docker for containerization.",
        is_relevant="No",
        question="What is the author's favorite food?",
        answer="I could not find relevant information in the uploaded documents."
    ).with_inputs("context", "question", "context_chunk"),

    dspy.Example(
        context="The KNN algorithm relies on distance metrics like Euclidean distance.",
        context_chunk="The KNN algorithm relies on distance metrics like Euclidean distance.",
        is_relevant="No",
        question="How did Hemanth use KNN?",
        answer="I could not find relevant information in the uploaded documents."
    ).with_inputs("context", "question", "context_chunk")
]

# --- 2. VALIDATION METRIC ---
def validate_answer(example, pred, trace=None):
    """
    Grades the LLM's practice answers during compilation.
    """
    # If we expect a refusal, the prediction MUST contain the refusal string
    if "I could not find relevant information" in example.answer:
        return "could not find" in pred.answer.lower()
    
    # Otherwise, check if the core concepts from the expected answer appear in the prediction
    expected_words = set(example.answer.lower().split())
    pred_words = set(pred.answer.lower().split())
    
    # Calculate word overlap (simple but effective for this scale)
    overlap = len(expected_words.intersection(pred_words)) / len(expected_words)
    return overlap > 0.4  # Pass if 40% of the key vocabulary is present

# --- 3. COMPILATION ---
def compile_rag():
    print("Loading unoptimized RAG Module...")
    student = RAGModule()
    
    print("Starting DSPy compilation. The LLM is practicing on the trainset...")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=validate_answer,
        max_bootstrapped_demos=3, # How many examples to inject into the final prompt
        max_labeled_demos=3,
        num_candidate_programs=5,
        num_threads=2
    )
    
    compiled_rag = optimizer.compile(student, trainset=trainset)
    
    # Save the new, balanced JSON
    save_path = "./data/compiled_rag.json"
    compiled_rag.save(save_path)
    print(f"Compilation complete! Your balanced model is saved to {save_path}")

if __name__ == "__main__":
    compile_rag()