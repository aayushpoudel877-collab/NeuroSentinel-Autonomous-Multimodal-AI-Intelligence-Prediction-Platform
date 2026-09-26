import re
def text_features(text):
    tokens=re.findall(r"[A-Za-z0-9']+",text.lower()); unique=len(set(tokens))
    return {'token_count':len(tokens),'unique_tokens':unique,'lexical_diversity':round(unique/max(len(tokens),1),4),'char_count':len(text)}
