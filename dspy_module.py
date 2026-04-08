import dspy
import os
import logging
import wikipedia
from config import settings

logger = logging.getLogger(__name__)

import json
import asyncio
from core.mcp_client import mcp_registry

# --- NEST_ASYNCIO FOR DSPY THREADS ---
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# --- PARALLEL MCP ROUTER ---
def parallel_mcp_executor(commands_json: str) -> str:
    """
    Execute multiple MCP tools concurrently to save time!
    Available tools: 'wikipedia_search' (kwargs: query), 'calculator' (kwargs: expression).
    Input MUST be a valid JSON list. Example:
    [{"name": "wikipedia_search", "kwargs": {"query": "Tokyo"}}, {"name": "calculator", "kwargs": {"expression": "2+2"}}]
    """
    try:
        commands = json.loads(commands_json)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(mcp_registry.execute_parallel(commands))
        
        output = []
        for cmd, res in zip(commands, results):
            output.append(f"Output of {cmd.get('name')}: {res}")
        return "\n".join(output)
    except Exception as e:
        return f"Parallel Execution Failed: {str(e)} (Make sure to pass a valid JSON list)"

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
    chat_history = dspy.InputField(desc="Previous conversation turns")
    answer = dspy.OutputField(desc="Factual answer grounded in context")

class QueryRouter(dspy.Signature):
    """
    Classify the incoming question into exactly one of these strategies:
    - 'fast': simple factual questions directly stated in the documents.
    - 'deep': complex questions requiring synthesis, summarization, or reasoning.
    - 'agentic': questions requiring real-time web search, current events, or math.
    - 'graph': questions about relationships, hierarchies, or connected entities.
    """
    question = dspy.InputField(desc="The user's query")
    route = dspy.OutputField(desc="Strictly one of: 'fast', 'deep', 'agentic', 'graph'")

class DeepSignature(dspy.Signature):
    """
    You are a Senior Expert Analyst. 
    Synthesize the provided context to answer complex questions.
    Think step-by-step to logically connect the information in the context.
    Use Chat History for continuity.
    """
    context = dspy.InputField(desc="Raw data snippets")
    question = dspy.InputField(desc="Complex query requiring synthesis")
    answer = dspy.OutputField(desc="Detailed report with reasoning")
    rationale = dspy.OutputField(desc="Reasoning steps (Chain of Thought)")

class AgentSignature(dspy.Signature):
    """
    You are an intelligent Agent that can use tools to find information or calculate answers.
    Answer the user's question. If you need facts, use wikipedia_search. If you need math, use calculator.
    """
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Final definitive answer")

class HallucinationGuard(dspy.Signature):
    """
    Verify if the 'answer' is fully supported by the 'context'.
    Output 'True' if supported, 'False' if it contains hallucinated info.
    """
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    is_supported = dspy.OutputField(desc="True or False")

class DocumentEvaluator(dspy.Signature):
    """
    Evaluate if the retrieved 'context_chunk' contains information relevant to answering the 'question'.
    Output exactly 'Yes' if relevant, or 'No' if completely irrelevant.
    """
    context_chunk = dspy.InputField(desc="A chunk of text from the knowledge base")
    question = dspy.InputField()
    is_relevant = dspy.OutputField(desc="Exactly 'Yes' or 'No'")

class EntityExtractor(dspy.Signature):
    """
    Extract key entities and their relationships from the given text.
    Output a list of relationships in format: Source | Relationship | Target
    Ensure there are spaces before and after the pipe character.
    If no relationships exist, output 'None'.
    """
    context_chunk = dspy.InputField(desc="Text to extract entities from")
    relationships = dspy.OutputField(desc="List of strings like 'EntityA | relates_to | EntityB'")

# --- 3. THE MODULE ---
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1. Rewriter (Memory)
        self.rewriter = dspy.ChainOfThought(RewriteSignature)
        
        # 2. Generators (QA)
        self.fast_prog = dspy.Predict(FastSignature)
        self.deep_prog = dspy.ChainOfThought(DeepSignature)
        
        # 3. Agentic RAG
        from dspy.predict.react import ReAct
        self.agent = ReAct(AgentSignature, tools=[parallel_mcp_executor])
        
        # 4. Router (Adaptive)
        self.router = dspy.Predict(QueryRouter)
        
        # 5. Guardrail (The "Critic")
        self.guard = dspy.Predict(HallucinationGuard)
        
        # 6. CRAG Evaluator
        self.doc_evaluator = dspy.Predict(DocumentEvaluator)
        
        # 7. Entity Extractor (GraphRAG)
        self.entity_extractor = dspy.Predict(EntityExtractor)
        
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

    def route_query(self, question):
        try:
            pred = self.router(question=question)
            route = pred.route.strip().lower()
            for valid in ["fast", "deep", "agentic", "graph"]:
                if valid in route:
                    return valid
            return "deep"
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return "deep"

    def evaluate_context(self, question, context_chunks):
        relevant = []
        for chunk in context_chunks:
            try:
                pred = self.doc_evaluator(context_chunk=chunk, question=question)
                if "yes" in pred.is_relevant.lower():
                    relevant.append(chunk)
            except Exception:
                relevant.append(chunk) # Default to keep if eval fails
        return relevant

    def forward(self, question, context, history_str="", mode="fast"):
        # 1. Generate Initial Answer
        if mode == "agentic":
            # For the agent, provide the existing context as part of the query so it can use tools if context is insufficient
            full_query = f"{question}\n\nContext from Knowledge Base:\n{context}"
            pred = self.agent(question=full_query)
            return pred
        elif mode == "deep":
            pred = self.deep_prog(context=context, 
                                  question=question, 
                                  chat_history=history_str)
            
            # 2. Apply Guardrail (Self-Correction) for Deep Mode
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                try:
                    # --- FAST NLI (QJL Signature Check) ---
                    # Using local QJL extraction as an ultra-fast NLI filter
                    try:
                        from core.quantization import QJLRetriever
                        from services.quantized_chroma import OllamaEmbeddingFunction
                        import numpy as np
                        
                        ef = OllamaEmbeddingFunction()
                        # Embed context and answer
                        ctx_emb = ef([context[:1000]])[0] # taking first 1k chars to avoid massive embedding
                        ans_emb = ef([pred.answer])[0]
                        
                        qjl = QJLRetriever(dim=768)
                        # We use 1-bit sign projections directly for fast cross-check without polar phase
                        ctx_jl = qjl._pack_bits(np.sign(np.dot(np.array(ctx_emb, dtype=np.float32), qjl.J)))
                        ans_jl = qjl._pack_bits(np.sign(np.dot(np.array(ans_emb, dtype=np.float32), qjl.J)))
                        
                        similarity = qjl.estimate_similarity(ctx_jl, ans_jl)
                        
                        # High similarity means 1-bit vectors align -> answer is grounded
                        if similarity > 0.6:
                            logger.info(f"Fast QJL NLI passed: {similarity}")
                            break # Validated fast path!
                    except Exception as e:
                        logger.warning(f"Fast QJL NLI skipped or failed: {e}")

                    # --- SLOW PATH NLI (CrossEncoder / LLM fallback) ---
                    check = self.guard(context=context, question=question, answer=pred.answer)
                    
                    if "false" not in check.is_supported.lower():
                        break # Validated!
                        
                    logger.warning(f"Guardrail flagged hallucination (Attempt {attempts+1}/{max_attempts}). Retrying...")
                    attempts += 1
                    
                    # Strictly instruct it to ground itself
                    if attempts < max_attempts:
                        stricter_question = f"{question} (Strictly base your answer ONLY on the context provided. Do not invent information.)"
                        pred = self.fast_prog(context=context, question=stricter_question)
                except Exception as e:
                    logger.error(f"Guardrail check failed: {e}")
                    break
                
            return pred
        else:
            return self.fast_prog(context=context, question=question, chat_history=history_str)