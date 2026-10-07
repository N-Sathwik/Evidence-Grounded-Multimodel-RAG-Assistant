from dataclasses import dataclass
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/'data'; DOCUMENT_DIR=DATA_DIR/'documents'; IMAGE_DIR=DATA_DIR/'images'; INDEX_DIR=DATA_DIR/'index'
@dataclass
class Settings:
    clip_model:str=os.getenv('CLIP_MODEL','openai/clip-vit-base-patch32')
    blip_model:str=os.getenv('BLIP_MODEL','Salesforce/blip-image-captioning-base')
    reranker_model:str=os.getenv('RERANKER_MODEL','cross-encoder/ms-marco-MiniLM-L6-v2')
    vlm_model:str=os.getenv('VLM_MODEL','Qwen/Qwen3-VL-2B-Instruct')
    chunk_size:int=int(os.getenv('CHUNK_SIZE','900')); chunk_overlap:int=int(os.getenv('CHUNK_OVERLAP','120'))
    top_k:int=int(os.getenv('TOP_K','10')); rerank_top_k:int=int(os.getenv('RERANK_TOP_K','5'))
settings=Settings()
for p in (DOCUMENT_DIR,IMAGE_DIR,INDEX_DIR): p.mkdir(parents=True,exist_ok=True)
