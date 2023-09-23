from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def ngram_vectorize(c, train_texts, train_labels, val_texts, ngram_range=(1,1)):
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'stop_words': "english",
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': c['TOKEN_MODE'], 
            'min_df': c['MIN_DOCUMENT_FREQUENCY'],
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)

    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(c['TOP_K'], x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32').todense()
    x_val = selector.transform(x_val).astype('float32').todense()
    return x_train, x_val, vectorizer.get_feature_names(), vectorizer