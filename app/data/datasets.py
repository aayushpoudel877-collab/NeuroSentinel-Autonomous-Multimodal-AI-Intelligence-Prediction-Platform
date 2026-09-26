from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class DatasetManifest:
    name:str
    version:str
    task:str
    source:str
    license:str='unspecified'
    notes:str=''
    def to_dict(self): return self.__dict__.copy()

def write_manifest(manifest,path='data/manifest.json'):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(manifest.to_dict(),indent=2),encoding='utf-8'); return str(target)
