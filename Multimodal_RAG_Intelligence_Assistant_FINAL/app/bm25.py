from rank_bm25 import BM25Okapi
class BM25Index:
    def __init__(self,records):
        self.records=records; self.index=BM25Okapi([str(r.get('text',r.get('caption',''))).lower().split() for r in records])
    def search(self,q,top_k=10):
        scores=self.index.get_scores(str(q).lower().split()); order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]
        return [{**self.records[i],'bm25_score':float(scores[i])} for i in order]
