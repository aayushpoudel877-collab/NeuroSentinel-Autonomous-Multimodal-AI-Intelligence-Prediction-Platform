from pathlib import Path
import json
def save_json(payload,path):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(payload,indent=2),encoding='utf-8'); return str(target)
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
