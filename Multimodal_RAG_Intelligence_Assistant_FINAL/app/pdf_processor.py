from pathlib import Path
import fitz, io
from PIL import Image

def extract_pdf(pdf_path,image_dir):
    pdf_path=Path(pdf_path); image_dir=Path(image_dir); image_dir.mkdir(parents=True,exist_ok=True)
    texts=[]; images=[]; doc=fitz.open(pdf_path)
    for page_no,page in enumerate(doc,start=1):
        text=page.get_text('text').strip()
        if text: texts.append({'document':pdf_path.name,'page':page_no,'text':text,'modality':'text'})
        for image_no,info in enumerate(page.get_images(full=True),start=1):
            raw=doc.extract_image(info[0]); ext=raw.get('ext','png')
            out=image_dir/f'{pdf_path.stem}_p{page_no}_img{image_no}.{ext}'; out.write_bytes(raw['image'])
            with Image.open(io.BytesIO(raw['image'])) as im: im.verify()
            images.append({'document':pdf_path.name,'page':page_no,'image_path':str(out),'modality':'image','image_id':f'{pdf_path.stem}_p{page_no}_img{image_no}'})
    return texts,images

def chunk_text(records,chunk_size=900,overlap=120):
    out=[]
    for rec in records:
        text=rec['text']; start=0
        while start<len(text):
            end=min(start+chunk_size,len(text)); out.append({**rec,'chunk_id':f"{rec['document']}:p{rec['page']}:c{len(out)}",'text':text[start:end]})
            if end==len(text): break
            start=max(0,end-overlap)
    return out
