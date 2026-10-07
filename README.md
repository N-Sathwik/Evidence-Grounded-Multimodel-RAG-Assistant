# Evidence-Grounded-Multimodel-RAG-Assistant
Evidence-grounded multimodal RAG system for text, images, charts and diagrams with hybrid retrieval, CLIP, BLIP, BM25, Cross-Encoder reranking, VLM generation and LangGraph orchestration.

# Multimodal RAG Intelligence Assistant

An evidence-grounded multimodal Retrieval-Augmented Generation
system for technical PDFs containing text, images, charts and diagrams.

## Features

- PDF text and image extraction
- Semantic chunking
- CLIP-based multimodal embeddings
- BLIP image captioning
- BM25 lexical retrieval
- Hybrid dense + lexical retrieval
- Cross-Encoder reranking
- Vision-Language Model generation
- Evidence-grounded responses
- Page-level citations
- LangGraph workflow orchestration
- Failure handling and fallbacks
- Retrieval evaluation using Hit@K

## Architecture

PDF
 ↓
Text + Images
 ↓
CLIP / BLIP
 ↓
Indexing
 ↓
Query
 ↓
CLIP + BM25
 ↓
Hybrid Retrieval
 ↓
Cross-Encoder
 ↓
Evidence
 ↓
Vision-Language Model
 ↓
Grounded Answer
 ↓
Citations

## Technologies

Python
PyMuPDF
CLIP
BLIP
BM25
Cross-Encoder
Vector Search
Transformers
LangGraph
PyTorch

## Installation

```bash
git clone <repository-url>
cd multimodal-rag-intelligence-assistant

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
