from dataclasses import dataclass,field
from typing import Any
@dataclass
class StepResult:
    name:str; ok:bool; data:Any=None; error:str|None=None; fallback_used:str|None=None
@dataclass
class Trace:
    steps:list[StepResult]=field(default_factory=list)
    def add(self,r): self.steps.append(r)
    def summary(self): return [{'step':s.name,'ok':s.ok,'error':s.error,'fallback':s.fallback_used} for s in self.steps]
def safe_call(name,fn,fallback=None):
    try:return StepResult(name,True,fn())
    except Exception as exc:
        if fallback is not None:
            try:return StepResult(name,False,fallback(),str(exc),'fallback')
            except Exception as fb:return StepResult(name,False,error=f'{exc}; fallback also failed: {fb}')
        return StepResult(name,False,error=str(exc))
