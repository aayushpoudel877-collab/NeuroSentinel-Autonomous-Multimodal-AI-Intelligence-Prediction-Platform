import re
import numpy as np

class HashingTextEmbedder:
    def __init__(self,dimensions=128): self.dimensions=dimensions
    def encode(self,text):
        vector=np.zeros(self.dimensions,dtype=np.float32)
        tokens=re.findall(r'[A-Za-z0-9]+',text.lower())
        for token in tokens:
            vector[hash(token)%self.dimensions]+=1.0
        norm=np.linalg.norm(vector)
        return (vector/norm if norm else vector).tolist()
    def encode_batch(self,texts): return [self.encode(t) for t in texts]
