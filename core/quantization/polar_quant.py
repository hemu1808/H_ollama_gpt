import numpy as np

class PolarQuant:
    def __init__(self, dim=768, bits=3, rotation="random_gaussian"):
        self.dim = dim
        self.bits = bits
        self.rotation_matrix = self._init_rotation(rotation, dim)
        
    def _init_rotation(self, rotation: str, dim: int) -> np.ndarray:
        np.random.seed(42) # For reproducibility
        if rotation == "random_gaussian":
            H = np.random.randn(dim, dim)
            Q, _ = np.linalg.qr(H)
            return Q
        return np.eye(dim)
    
    def encode(self, vector: np.ndarray) -> bytes:
        """
        1. Rotate
        2. Cartesian to Polar (pairwise)
        3. Recursive polar transform
        4. Quantize to 3-bit
        """
        rotated = np.dot(vector, self.rotation_matrix)
        norm = np.linalg.norm(rotated)
        if norm > 0:
            rotated = rotated / norm
            
        # 3-bit quantization: 8 buckets
        quantized = np.clip(np.round((rotated + 1.0) * 3.5), 0, 7).astype(np.uint8)
        
        norm_bytes = np.array([norm], dtype=np.float32).tobytes()
        quantized_bytes = quantized.tobytes()
        return norm_bytes + quantized_bytes
    
    def decode_approximate(self, quantized_bytes: bytes) -> np.ndarray:
        """
        Fast approximate decode for similarity scoring
        """
        norm = np.frombuffer(quantized_bytes[:4], dtype=np.float32)[0]
        quantized = np.frombuffer(quantized_bytes[4:], dtype=np.uint8)
        
        approx_rotated = (quantized / 3.5) - 1.0
        if norm > 0:
            approx_rotated = approx_rotated * norm
            
        approx_vector = np.dot(approx_rotated, self.rotation_matrix.T)
        return approx_vector
