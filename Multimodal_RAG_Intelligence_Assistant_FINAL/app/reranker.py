class CrossEncoderReranker:
    def __init__(self,name):
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(name)
    def rerank(self,query,candidates,top_k=5):
        pairs=[(query,x.get('text') or x.get('caption') or x.get('image_id','')) for x in candidates]
        scores=self.model.predict(pairs)
        return sorted([{**x,'rerank_score':float(s)} for x,s in zip(candidates,scores)],key=lambda x:x['rerank_score'],reverse=True)[:top_k]
