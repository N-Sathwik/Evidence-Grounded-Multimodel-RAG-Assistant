import torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor,BlipForConditionalGeneration,BlipProcessor
class CLIPEncoder:
    def __init__(self,name='openai/clip-vit-base-patch32'):
        self.processor=CLIPProcessor.from_pretrained(name); self.model=CLIPModel.from_pretrained(name); self.model.eval()
    def _pool(self,o):
        if hasattr(o,'pooler_output') and o.pooler_output is not None:return o.pooler_output
        if torch.is_tensor(o):return o
        raise TypeError(type(o))
    def encode_text(self,texts):
        x=self.processor(text=texts,return_tensors='pt',padding=True,truncation=True)
        with torch.no_grad(): f=self.model.text_projection(self._pool(self.model.text_model(**x)))
        return torch.nn.functional.normalize(f,dim=-1).cpu().numpy()
    def encode_images(self,paths):
        ims=[Image.open(p).convert('RGB') for p in paths]; x=self.processor(images=ims,return_tensors='pt')
        with torch.no_grad(): f=self.model.visual_projection(self._pool(self.model.vision_model(pixel_values=x['pixel_values'])))
        return torch.nn.functional.normalize(f,dim=-1).cpu().numpy()
    def encode_query(self,text): return self.encode_text([text])[0]
class BLIPCaptioner:
    def __init__(self,name='Salesforce/blip-image-captioning-base'):
        self.processor=BlipProcessor.from_pretrained(name); self.model=BlipForConditionalGeneration.from_pretrained(name); self.model.eval()
    def caption(self,path):
        x=self.processor(images=Image.open(path).convert('RGB'),return_tensors='pt')
        with torch.no_grad(): ids=self.model.generate(**x,max_new_tokens=60)
        return self.processor.decode(ids[0],skip_special_tokens=True)
class LocalQwenVLM:
    def __init__(self,name):
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        self.processor=AutoProcessor.from_pretrained(name)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(name,torch_dtype='auto',device_map='auto')
    def answer(self,question,text_evidence,image_paths):
        content=[{'type':'text','text':question}]
        for p in image_paths: content.append({'type':'image','image':str(p)})
        content.append({'type':'text','text':'Use only supplied evidence. If insufficient, say so.\n\n'+'\n\n'.join(text_evidence)})
        messages=[{'role':'user','content':content}]
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        kwargs={'text':[prompt],'return_tensors':'pt','padding':True}
        if image_paths: kwargs['images']=[Image.open(p).convert('RGB') for p in image_paths]
        x=self.processor(**kwargs); x={k:(v.to(self.model.device) if hasattr(v,'to') else v) for k,v in x.items()}
        with torch.no_grad(): out=self.model.generate(**x,max_new_tokens=500)
        trimmed=[o[len(i):] for i,o in zip(x['input_ids'],out)]
        return self.processor.batch_decode(trimmed,skip_special_tokens=True)[0]
