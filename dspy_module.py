import dspy
import os
from config import settings

# --- 1. CONFIGURATION ---
def get_llm():
    # We use a slightly higher temperature for the teacher during training
    # but keep it low for inference to be factual.
    return dspy.LM(
        model='ollama/' + settings.OLLAMA_LLM_MODEL, 
        api_base=settings.OLLAMA_URL,
        api_key=""
    )

try:
    dspy.configure(lm=get_llm())
except Exception as e:
    print(f"Error configuring DSPy: {e}")

# --- 2. SIGNATURES (PROMPTS) ---

class RewriteSignature(dspy.Signature):
    """
    You are a Search Query Optimizer for a Resume/Portfolio RAG system.
    1. Read the "Chat History" to understand the user's intent.
    2. Rewrite the "Follow_up Question" into a SPECIFIC, STANDALONE search query.
    3. DISAMBIGUATE pronouns (it, that, he, the project) using the history.
    4. If the question is already specific, output it unchanged.
    """
    chat_history = dspy.InputField(desc="Previous conversation turns")
    follow_up_question = dspy.InputField(desc="The user's current question")
    standalone_query = dspy.OutputField(desc="The specific search query")

class FastSignature(dspy.Signature):
    """
    You are an expert Technical Recruiter Assistant.
    Answer the question based ONLY on the provided context about Hemanth.
    
    CRITICAL RULES:
    1. GROUNDING: If the answer is not in the context, say "I could not find relevant information in the uploaded documents."
    2. ANTI-DICTIONARY: Do not define technical terms (e.g., don't explain what "Docker" is). Explain how Hemanth USED them.
    3. SPECIFICITY: Be precise about metrics, dates, and technologies mentioned in the text.
    4. STYLE: Professional, concise, and direct.
    """
    context = dspy.InputField(desc="Relevant snippets from Hemanth's Resume/Docs")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Factual answer grounded in context")

class DeepSignature(dspy.Signature):
    """
    You are a Senior Technical Analyst. 
    Synthesize the provided context to answer complex questions.
    Think step-by-step about the relationships between projects, skills, and timeline.
    """
    context = dspy.InputField(desc="Raw data snippets")
    question = dspy.InputField(desc="Complex query requiring synthesis")
    answer = dspy.OutputField(desc="Detailed report with reasoning")

# --- 3. THE MODULE ---
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # The Rewriter (Handles Memory)
        self.rewriter = dspy.ChainOfThought(RewriteSignature)
        
        # The Generators (Handle QA)
        self.fast_prog = dspy.Predict(FastSignature)
        self.deep_prog = dspy.ChainOfThought(DeepSignature)
        
        # Load optimized weights (The "Training" Result)
        self._load_compiled_program()

    def _load_compiled_program(self):
        """Attempts to load the 'Brain' optimized by train_dspy.py"""
        compiled_path = "./data/compiled_rag.json"
        if os.path.exists(compiled_path):
            try:
                self.load(compiled_path)
                print(f"Loaded optimized DSPy program from {compiled_path}")
            except Exception as e:
                print(f"Failed to load optimized program: {e}")

    def rewrite_query(self, question, history_str):
        """
        Turns 'tell me more about it' -> 'Details about the Orchestration Engine'
        """
        # If no history, no need to rewrite
        if not history_str:
            return question
            
        pred = self.rewriter(chat_history=history_str, follow_up_question=question)
        return pred.standalone_query

    def forward(self, question, context, mode="fast"):
        if mode == "deep":
            return self.deep_prog(context=context, question=question)
        else:
             prediction = self.fast_prog(context=context, question=question)
             dspy.Suggest(
                 len(prediction.answer) < 500,
                 "The answer is too long. Summarize it to be under 500 characters.")
             dspy.Suggest(
              "context" in prediction.answer.lower() or "document" in prediction.answer.lower() or len(context) > 0,
              "The answer must be derived from the context provided."
              )
             return prediction