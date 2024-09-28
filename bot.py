import random
import numpy as np
import pickle
import json

import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import string 
import re

from tensorflow.keras.models import load_model


model = load_model("chatbot_model.h5")
intents = json.loads(open("intent.json", encoding="utf8").read())
words = pickle.load(open("words.pkl", "rb"))
clss = pickle.load(open("classes.pkl", "rb"))


list_stopwords = set(stopwords.words('english'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def chatbot_message(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def preprocess_sentence(sentence):
    sentence_words = sentence.lower()
    sentence_words = sentence.translate(str.maketrans("","",string.punctuation))
    sentence_words = sentence.strip()
    sentence_words = re.sub('\s+',' ',sentence)    

    sentence_words = word_tokenize(sentence)

    words = [w for w in sentence_words if not w in list_stopwords]

    sentence_words = [stemmer.stem(w) for w in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):

    sentence_words = preprocess_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["intent"] == tag:
            result = random.choice(i["responses"])
            return result

def chat(inp):
    x=chatbot_message(inp)
    out = str(x)
    return out
    
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
       
        return_list.append({"intent": clss[r[0]], "probability": str(r[1])})
    return return_list


