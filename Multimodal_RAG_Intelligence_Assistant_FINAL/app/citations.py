def source_label(x): return f"{x.get('document','unknown')} — page {x.get('page','?')} — {x.get('modality','unknown')}"
def collect_sources(items):
    seen=set(); out=[]
    for x in items:
        k=(x.get('document'),x.get('page'),x.get('modality'),x.get('image_id'))
        if k in seen:continue
        seen.add(k); out.append({'document':x.get('document'),'page':x.get('page'),'modality':x.get('modality'),'image_path':x.get('image_path'),'label':source_label(x)})
    return out
def evidence_only_answer(question,items):
    if not items:return {'status':'partial_failure','answer':'No usable evidence was retrieved for this question.','sources':[]}
    lines=[f'Evidence available for: {question}','']
    for x in items:
        lines.append(f"[{source_label(x)}] {x.get('text') or x.get('caption') or 'visual evidence available'}")
    return {'status':'evidence_only','answer':'\n'.join(lines),'sources':collect_sources(items)}
