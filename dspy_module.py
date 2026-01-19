import dspy
import os
import logging
from config import settings

logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
def get_llm():
    return dspy.LM(
        model='ollama/' + settings.OLLAMA_LLM_MODEL, 
        api_base=settings.OLLAMA_URL,
        api_key="ollama",
        temperature=0.1,
        num_ctx = 4096  # Low temp for factual consistency
    )

try:
    dspy.configure(lm=get_llm())
except Exception as e:
    logger.error(f"Error configuring DSPy: {e}")

# --- 2. SIGNATURES ---

class RewriteSignature(dspy.Signature):
    """
    You are a Search Query Optimizer.
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
    You are an expert RAG Assistant. 
    Answer the question based STRICTLY on the provided 'Context'.
    
    CRITICAL RULES:
    1. GROUNDING: If the answer is not in the context, say "I could not find relevant information in the uploaded documents."
    2. STYLE: Professional, concise, and direct.
    3. Do NOT use outside knowledge or training examples. 
    4. Use 'Chat History' to understand the conversation context.
    """
    context = dspy.InputField(desc="Relevant snippets from documents")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Factual answer grounded in context")

class DeepSignature(dspy.Signature):
    """
    You are a Senior Expert Analyst. 
    Synthesize the provided context to answer complex questions.
    Think step-by-step about the relationships between projects, skills, and timeline.
    Use Chat History for continuity.
    """
    context = dspy.InputField(desc="Raw data snippets")
    question = dspy.InputField(desc="Complex query requiring synthesis")
    answer = dspy.OutputField(desc="Detailed report with reasoning")
    rationale = dspy.OutputField(desc="Reasoning steps (Chain of Thought)")

class HallucinationGuard(dspy.Signature):
    """
    Verify if the 'answer' is fully supported by the 'context'.
    Output 'True' if supported, 'False' if it contains hallucinated info.
    """
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    is_supported = dspy.OutputField(desc="True or False")

# --- 3. THE MODULE ---
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1. Rewriter (Memory)
        self.rewriter = dspy.ChainOfThought(RewriteSignature)
        
        # 2. Generators (QA)
        self.fast_prog = dspy.Predict(FastSignature)
        self.deep_prog = dspy.ChainOfThought(DeepSignature)
        
        # 3. Guardrail (The "Critic")
        self.guard = dspy.Predict(HallucinationGuard)
        
        # Load optimized weights
        self._load_compiled_program()

    def _load_compiled_program(self):
        compiled_path = "./data/compiled_rag.json"
        if os.path.exists(compiled_path):
            try:
                self.load(compiled_path)
                logger.info(f"Loaded optimized DSPy program from {compiled_path}")
            except Exception as e:
                logger.warning(f"Failed to load optimized program: {e}")

    def rewrite_query(self, question, history_str):
        if not history_str:
            return question
        try:
            pred = self.rewriter(chat_history=history_str, follow_up_question=question)
            return pred.standalone_query
        except Exception:
            return question

    def forward(self, question, context, history_str="", mode="fast"):
        # 1. Generate Initial Answer
        if mode == "deep":
            pred = self.deep_prog(context=context, 
                                  question=question, 
                                  chat_history=history_str)
            
            # 2. Apply Guardrail (Self-Correction) for Deep Mode
            try:
                check = self.guard(context=context, question=question, answer=pred.answer)
                
                # If the guard says "False" (unsupported), we retry with a stricter prompt
                if "false" in check.is_supported.lower():
                    logger.warning("Guardrail flagged hallucination. Retrying...")
                    
                    # Retry using the Fast signature (which is stricter) or re-prompting
                    pred = self.fast_prog(
                        context=context, 
                        question=f"{question} (Strictly base your answer ONLY on the context provided.)"
                    )
            except Exception as e:
                logger.error(f"Guardrail check failed: {e}")
                
            return pred
        else:
            return self.fast_prog(context=context, question=question, chat_history=history_str)