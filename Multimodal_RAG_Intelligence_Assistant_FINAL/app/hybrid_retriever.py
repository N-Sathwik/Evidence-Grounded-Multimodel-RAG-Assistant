def _norm(vals):
    if not vals:return {}
    lo,hi=min(vals),max(vals)
    return {i:(v-lo)/(hi-lo) if hi>lo else 0.0 for i,v in enumerate(vals)}
def fuse_results(dense,sparse,alpha=.65,top_k=10):
    out={}; dn=_norm([x.get('dense_score',0) for x in dense]); sn=_norm([x.get('bm25_score',0) for x in sparse])
    for i,x in enumerate(dense):
        k=x.get('chunk_id') or x.get('image_id') or x.get('text','')[:80]; out.setdefault(k,dict(x)); out[k]['fused_score']=out[k].get('fused_score',0)+alpha*dn[i]
    for i,x in enumerate(sparse):
        k=x.get('chunk_id') or x.get('image_id') or x.get('text','')[:80]; out.setdefault(k,dict(x)); out[k]['fused_score']=out[k].get('fused_score',0)+(1-alpha)*sn[i]
    return sorted(out.values(),key=lambda x:x.get('fused_score',0),reverse=True)[:top_k]
