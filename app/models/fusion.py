class MultimodalFusion:
    def combine(self,signals):
        usable=[s for s in signals if 'score' in s]
        if not usable: return {'score':0.0,'confidence':0.0,'sources':0}
        weights=[max(0.0,float(s.get('confidence',1.0))) for s in usable]; total=sum(weights) or 1.0
        score=sum(float(s['score'])*w for s,w in zip(usable,weights))/total
        return {'score':round(score,6),'confidence':round(min(1.0,sum(weights)/len(usable)),6),'sources':len(usable)}
