##########################################################
# THIS CODE RUNS WITH ANACONDA3 AS THE INTERPRETER
# USES PYTHON 3.6 NOT 3.7 DUE TO TFLEARN NOT WORKING PROPERLY ON 3.7
# ONCE CHATBOT IS WORKING PROPERLY, GUI AND TEXT TO SPEECH CODES WILL BE MERGED INTO ONE
##########################################################
import numpy
import random
import tflearn
import tensorflow
import pickle

# Importing json to be able to read json format files
# Using json as our data file format, used as a dictionary
import json

# Using LancasterStemmer from the nltk library to stem words
import nltk
from nltk.stem.lancaster import LancasterStemmer
# Labeling the LancasterStemmer function as stem
stemming = LancasterStemmer()

# Opening the outcomes.json file and labeling it as our data
with open("outcomes.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        word, label, train, output = pickle.load(f)

except:
        docX = []
        docY = []
        label = []
        word = []

        # For every outcome in the data
        for outcomes in data["outcomes"]:
            # For every input in outcomes
            for inputs in outcomes["inputs"]:
                # Grabs all the words within the inputs and turns them into tokens
                words = nltk.word_tokenize(inputs)
                # Extending the list of words instead of appending each one in
                word.extend(words)
                # Adding to docX the input words
                docX.append(words)
                # Adding to docY the titles so we can better classify
                docY.append(outcomes["title"])
            # Gets all the titles
            if outcomes ["title"] not in label:
                label.append(outcomes["title"])

        # Sorts the labels
        label = sorted(label)

        # Removes duplicate elements
        word = sorted(list(set(word)))

        # Stemming word down to the root word(example: words = word)
        # This makes the bot more accurate
        # This will also makes all the letters lower case
        # Also removes question marks
        word = [stemming.stem(w.lower()) for w in word if w != "?"]

        # Creating a bag of words for our neural network

        train = []
        output = []

        # Sets a zero for each word that does not exist
        output_empty = [0 for _ in range(len(label))]

        for x, doc in enumerate(docX):
            word_bag = []
            # Stemming words down to the root word
            words = [stemming.stem(w) for w in doc]

            # If the word exists we put a 1, if it doesn't exist we put a 0
            for w in word:
                if w in words:
                    word_bag.append(1)
                else:
                    word_bag.append(0)

                # Creates a copy
                output_row = output_empty[:]

                # Look through label list and get the title to set it to 1 in the output row
                output_row[label.index(docY[x])] = 1

                # Append the list
                train.append([word_bag])
                # Append the output row
                output.append(output_row)

        # Turning the train data to numpy arrays
        train = numpy.array(train)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((word, label, train, output), f)

tensorflow.reset_default_graph()

network = tflearn.input_data(shape=[None, len(train[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation="softmax")
net = tflearn.regression(network)

model = tflearn.DNN(network)

try:
    model.load("model.tflearn")
except:
    model.fit(train, output, n_epoch=1000,  batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, word):
    word_bag = [0]

    s_word = nltk.word_tokenize(s)
    s_word = [stemming.stem(wrd.lower()) for wrd in s_word]

    for se in s_word:
        for i, w in enumerate(word):
            if w == se:
                word_bag[i].append(1)

    return numpy.array(word_bag)

def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, word)])






