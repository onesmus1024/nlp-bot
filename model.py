import random
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS

vocal_size=2000
max_len=None
label_len=None
x_train=None
y_train=None
model=None
active_model=None
tag=[]
questions=[]
responses={}
tokenizer = Tokenizer(num_words=vocal_size,oov_token='<oov>')
le=LabelEncoder()

def get_data():
    global max_len
    global label_len
    global x_train
    global y_train
    global model
    global tag
    global questions
    global responses
    with open("data/intent.json") as content:
        data = json.load(content)
    # extract data from json file
    for intent in data['intents']:
        responses[intent['intent']]=intent['responses']
        for question in intent['text']:
            tag.append(intent['intent'])
            questions.append(question)

    # encode the tag and questions
    df = pd.DataFrame({'intent':tag,'question':questions})
    y_train=df['intent']
    x_train=df['question']
    y_train=le.fit_transform(y_train)
    tokenizer.fit_on_texts(df['question'])
    x_train=tokenizer.texts_to_sequences(x_train)
    x_train=sequence.pad_sequences(x_train,maxlen=10)
    max_len=max([len(k) for k in x_train])
    print(max_len)
    label_len=len(set(y_train))
    print(label_len)


def create_model():
    global model 
    model = keras.Sequential([
    layers.Input(max_len),
    layers.Embedding(vocal_size,10),
    layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Flatten(),
    layers.Dense(label_len,activation='softmax')
    ])

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = model.fit(x_train,y_train,epochs=400)
    model.save("./saved_model")



# plt.plot(history.history['loss'],label='loss')
# plt.plot(history.history['accuracy'],label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


def predict(text):
    global active_model
    if active_model is None:
        active_model = tf.keras.models.load_model('./saved_model')
    text=tokenizer.texts_to_sequences([text])
    text=sequence.pad_sequences(text,maxlen=max_len)
    pred=active_model.predict(text)
    pred=pred.argmax()
    tag= le.inverse_transform([pred])[0]
    response = random.choice(responses[tag])
    return response



