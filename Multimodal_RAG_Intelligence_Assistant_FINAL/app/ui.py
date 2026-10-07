import json,streamlit as st
from app.config import settings,INDEX_DIR
from app.models import CLIPEncoder,LocalQwenVLM
from app.vector_store import NumpyVectorStore
from app.bm25 import BM25Index
from app.reranker import CrossEncoderReranker
from app.pipeline import MultimodalRAG
st.set_page_config(page_title='Multimodal RAG Intelligence Assistant',layout='wide'); st.title('Multimodal RAG Intelligence Assistant'); st.caption('Evidence-grounded PDF Q&A with text + image retrieval')
@st.cache_resource
def load_rag():
    store=NumpyVectorStore(INDEX_DIR); store.load(); rec=json.loads((INDEX_DIR/'bm25_records.json').read_text()); clip=CLIPEncoder(settings.clip_model); bm=BM25Index(rec)
    try:rr=CrossEncoderReranker(settings.reranker_model)
    except Exception:rr=None
    try:vlm=LocalQwenVLM(settings.vlm_model)
    except Exception:vlm=None
    return MultimodalRAG(clip,store,bm,rr,vlm,settings.top_k,settings.rerank_top_k)
q=st.text_input('Ask a question about indexed PDFs')
if q:
    try:
        r=load_rag().answer(q); st.subheader('Answer'); st.write(r['answer']); st.subheader('Evidence'); [st.write(x['label']) for x in r.get('sources',[])];
        with st.expander('Execution trace'): st.json(r.get('trace',[]))
    except Exception as e: st.error(f'Request failed safely: {e}')
