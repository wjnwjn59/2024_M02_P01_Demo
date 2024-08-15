import string
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from sklearn.preprocessing import LabelEncoder

def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    return tokens

def create_dictionary(messages):
    dictionary = []

    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary

def encode_label(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return y, le

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features

def run_data_preprocess(df):
    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()

    messages = [preprocess_text(message) for message in messages]
    dictionary = create_dictionary(messages)
    X = np.array([create_features(tokens, dictionary) for tokens in messages])
    y, le = encode_label(labels)

    return X, y, dictionary, le