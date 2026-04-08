import numpy as np

class QJLRetriever:
    def __init__(self, dim=768, jl_dim=256):
        # Johnson-Lindenstrauss transform matrix
        np.random.seed(42)  # For deterministic JL projection
        self.J = np.random.randn(dim, jl_dim) / np.sqrt(jl_dim)
    
    def encode_residual(self, original: np.ndarray, polar_approx: np.ndarray) -> bytes:
        """1-bit sign of JL projection of error"""
        residual = original - polar_approx
        projected = np.sign(np.dot(residual, self.J))
        return self._pack_bits(projected)
    
    def _pack_bits(self, bit_array: np.ndarray) -> bytes:
        binary = (bit_array > 0).astype(np.uint8)
        return np.packbits(binary).tobytes()
        
    def estimate_similarity(self, query_jl_signs: bytes, residual_jl_signs: bytes) -> float:
        """Zero-overhead similarity estimation (Hamming distance approximation)"""
        q_bits = np.unpackbits(np.frombuffer(query_jl_signs, dtype=np.uint8))
        r_bits = np.unpackbits(np.frombuffer(residual_jl_signs, dtype=np.uint8))
        
        matches = np.sum(q_bits == r_bits)
        total_bits = len(q_bits)
        similarity = np.cos(np.pi * (1.0 - (matches / total_bits)))
        return float(similarity)
