import silence_tensorflow.auto
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import pickle

import json
with open('V3/intents.json') as file:
    data = json.load(file)

try:  
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
    # Data preprocessing        

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

#    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w.lower()) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Deeplearn Model

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Starten Sie ein Gespräch mit dem Bot ('Stop' um zu beenden)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "stop":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        max_wahrsch = numpy.array(results)
        max_value = numpy.max(max_wahrsch)
#       print(max_value)
        if max_value < 0.76:   
            zufallsantworten = ["Oh wirklich...", "Interessant", "Das kann man so sehen", "LOL Wut?", "Hä? :/",
                                "Ich verstehe.......nicht XD", "Mmhh", "Naja", "Wovon sprechen Sie?", "Bitte nehmen Sie die Hände von der Tastatur"]
            print(random.choice(zufallsantworten))
            continue

        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()