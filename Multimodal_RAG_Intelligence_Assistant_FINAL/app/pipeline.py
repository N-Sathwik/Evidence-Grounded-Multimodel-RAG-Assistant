from .citations import collect_sources,evidence_only_answer
from .hybrid_retriever import fuse_results
from .reliability import Trace,StepResult
class MultimodalRAG:
    def __init__(self,clip,store,bm25,reranker=None,vlm=None,top_k=10,rerank_top_k=5): self.clip=clip; self.store=store; self.bm25=bm25; self.reranker=reranker; self.vlm=vlm; self.top_k=top_k; self.rerank_top_k=rerank_top_k
    def retrieve(self,q):
        trace=Trace(); dense=[]; sparse=[]
        try:dense=self.store.search(self.clip.encode_query(q),self.top_k); trace.add(StepResult('dense_retrieval',True,dense))
        except Exception as e:trace.add(StepResult('dense_retrieval',False,error=str(e),fallback_used='BM25'))
        try:sparse=self.bm25.search(q,self.top_k); trace.add(StepResult('bm25_retrieval',True,sparse))
        except Exception as e:trace.add(StepResult('bm25_retrieval',False,error=str(e),fallback_used='dense'))
        candidates=fuse_results(dense,sparse,top_k=self.top_k) if dense and sparse else sorted(dense or sparse,key=lambda x:x.get('dense_score',x.get('bm25_score',0)),reverse=True)[:self.top_k]
        trace.add(StepResult('fusion',bool(candidates),candidates,error=None if candidates else 'No candidates')); return candidates,trace
    def answer(self,q,use_vlm=True):
        candidates,trace=self.retrieve(q)
        if not candidates:return {'status':'partial_failure','answer':'No evidence was retrieved. Re-index or reformulate the query.','sources':[],'trace':trace.summary()}
        if self.reranker:
            try:candidates=self.reranker.rerank(q,candidates,self.rerank_top_k); trace.add(StepResult('reranking',True,candidates))
            except Exception as e:trace.add(StepResult('reranking',False,error=str(e),fallback_used='fused ranking'))
        verified=[x for x in candidates if x.get('text') or x.get('image_path') or x.get('caption')]
        text=[f"{x.get('document')} page {x.get('page')}: {x.get('text','')[:1200]}" for x in verified if x.get('text')]
        images=[x['image_path'] for x in verified if x.get('image_path')][:3]
        if use_vlm and self.vlm:
            try:
                ans=self.vlm.answer(q,text,images); trace.add(StepResult('grounded_generation',True,ans)); return {'status':'success','answer':ans,'sources':collect_sources(verified),'trace':trace.summary()}
            except Exception as e:trace.add(StepResult('grounded_generation',False,error=str(e),fallback_used='evidence_only'))
        out=evidence_only_answer(q,verified); out['trace']=trace.summary(); return out
