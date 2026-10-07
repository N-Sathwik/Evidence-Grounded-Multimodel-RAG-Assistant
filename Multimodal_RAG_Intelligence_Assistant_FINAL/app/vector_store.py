from pathlib import Path
import json,numpy as np
class NumpyVectorStore:
    def __init__(self,path): self.path=Path(path); self.embeddings=None; self.records=[]
    def add(self,embeddings,records): self.embeddings=np.asarray(embeddings,dtype=np.float32); self.records=list(records)
    def search(self,q,top_k=10):
        if self.embeddings is None or not self.records:return []
        q=np.asarray(q,dtype=np.float32); q=q/(np.linalg.norm(q)+1e-12); scores=self.embeddings@q; idx=np.argsort(-scores)[:top_k]
        return [{**self.records[i],'dense_score':float(scores[i])} for i in idx]
    def save(self): self.path.mkdir(parents=True,exist_ok=True); np.save(self.path/'embeddings.npy',self.embeddings); (self.path/'records.json').write_text(json.dumps(self.records,indent=2),encoding='utf-8')
    def load(self): self.embeddings=np.load(self.path/'embeddings.npy'); self.records=json.loads((self.path/'records.json').read_text(encoding='utf-8'))
