from pathlib import Path
import json
def load_questions(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def evaluate_hit_at_k(rows,k=5):
    hits=0
    for r in rows:
        expected=set(tuple(x) for x in r['expected_sources']); got={(x.get('document'),x.get('page')) for x in r['retrieved'][:k]}
        hits+=bool(expected&got)
    return {'hit_at_k':hits/len(rows) if rows else 0.0,'questions':len(rows)}
