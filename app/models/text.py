import re
from collections import Counter

POSITIVE = {"good","great","excellent","success","successful","happy","safe","improve","improved","positive","strong","efficient"}
NEGATIVE = {"bad","poor","failure","failed","sad","unsafe","risk","danger","negative","weak","slow","problem"}

class TextAnalyzer:
    def analyze(self, text: str):
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        counts = Counter(tokens)
        pos = sum(counts[w] for w in POSITIVE)
        neg = sum(counts[w] for w in NEGATIVE)
        score = (pos - neg) / max(len(tokens), 1)
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        return {
            "task": "sentiment_analysis",
            "label": label,
            "score": round(float(score), 4),
            "positive_terms": pos,
            "negative_terms": neg,
            "tokens": len(tokens)
        }
