import re
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

def normalize(text):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])
    filtered_sentence = []
    for w in text.split(' '): 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)
