from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone

class ArtifactStore:
    def __init__(self,root='models/artifacts'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def save_metadata(self,name,payload):
        record=dict(payload); record['name']=name; record['created_at']=datetime.now(timezone.utc).isoformat(); record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        target=self.root/f'{name}.json'; target.write_text(json.dumps(record,indent=2),encoding='utf-8'); return record
    def load_metadata(self,name):
        return json.loads((self.root/f'{name}.json').read_text(encoding='utf-8'))
