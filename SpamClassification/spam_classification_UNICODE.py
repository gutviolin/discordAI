from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import random
import string
import numpy as np
import os

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

letter_index = {}
hm_lines = 10000000

def generate_spam():
    dict = get_letter_dict_reversed()
    with open("Spam.txt", "w") as text_file:
        #10000 lines of spam
        for i in range(10000):
            line = ""
            for j in range(random.randint(1,256)):
                line += dict[random.randint(0,len(dict)-1)]
            text_file.write(line +'\n')

def get_letter_dict():
    with open('unicode_no_num.txt', 'r', encoding="utf-8") as file:
        array_of_content = [line.splitlines() for line in file]
        dictOfWords = { i : array_of_content[i] for i in range(0, len(array_of_content) ) }
        print (dictOfWords)
    return dictOfWords

def get_letter_dict_reversed():
    letter_index_temp = get_letter_dict()
    reverse_letter_index = dict([(value, key) for (key, value) in letter_index_temp.items()])
    return reverse_letter_index

def get_letter_dict_with_reserved():
    letter_index = get_letter_dict()
    letter_index = {k:(v+4) for k,v in letter_index.items()}
    letter_index["<PAD>"] = 0
    letter_index["<START>"] = 1
    letter_index["<UNK>"] = 2  # unknown
    letter_index["<UNUSED>"] = 3
    return letter_index

def get_letter_dict_with_reserved_reversed():
    letter_index_temp = get_letter_dict_with_reserved()
    reverse_letter_index = dict([(value, key) for (key, value) in letter_index_temp.items()])
    return reverse_letter_index

reverse_letter_index = get_letter_dict_with_reserved_reversed()
letter_index = get_letter_dict_with_reserved()

def decode(text):
    return ''.join([reverse_letter_index.get(i, 2) for i in text])

def encode(text):
    chars = list(text)
    return ([letter_index.get(i, 2) for i in chars])

def encode_contents(not_spam, spam):
    contents_of_file_encoded = []
    with open(not_spam,'r') as f:
        contents_of_file = [line.splitlines() for line in f]
        for i in range(len(contents_of_file)):
            contents_of_file_encoded.append(encode(contents_of_file[i][0]))

    with open(spam,'r') as g:
        contents_of_file = [line.splitlines() for line in g]
        for j in range(len(contents_of_file)):
            contents_of_file_encoded.append(encode(contents_of_file[j][0]))

    return contents_of_file_encoded

def encode_contents_single(file):
    contents_of_file_encoded = []
    with open(file,'r') as f:
        contents_of_file = [line.splitlines() for line in f]
        for i in range(len(contents_of_file)):
            contents_of_file_encoded.append(encode(contents_of_file[i][0]))
    return contents_of_file_encoded

def suffle_contents(spam_and_not, file1, file2):
    not_spam = encode_contents_single(file1)
    spam = encode_contents_single(file2)
    total_file_data = []
    for i in range(len(not_spam)):
        total_file_data.append(1)
    for j in range(len(spam)):
        total_file_data.append(0)

    test = (spam_and_not)
    rng_state = np.random.get_state()
    np.random.shuffle(test)
    np.random.set_state(rng_state)
    np.random.shuffle(total_file_data)

    return test, total_file_data

file_contents_encoded = encode_contents("NotSpam.txt", "Spam.txt")

data, labels = suffle_contents(file_contents_encoded, "NotSpam.txt", "Spam.txt")

train_data_pre = data[:len(data)//2]
test_data_pre = data[len(data)//2:]
train_data = np.array(train_data_pre)
test_data = np.array(test_data_pre)

train_labels_pre = labels[:len(labels)//2]
test_labels_pre = labels[len(labels)//2:]
train_labels = np.array(train_labels_pre)
test_labels = np.array(test_labels_pre)


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=letter_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=letter_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

def build_model():
    vocab_size = len(get_letter_dict_with_reserved())

    model = tf.keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1
                    )

    results = model.evaluate(test_data, test_labels)

    model.fit(train_data, train_labels,  epochs = 10,
                validation_data = (test_data,test_labels),
                callbacks = [cp_callback])  # pass callback to training

    print(results)
    return model

def create_default_model():
    vocab_size = len(get_letter_dict_with_reserved())
    model = tf.keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

    return model

def create_restored_model():
    restore_model = create_default_model()
    restore_model.load_weights(checkpoint_path)
    loss,acc = restore_model.evaluate(test_data, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    return restore_model

def test_data_msg(message):
    pre = np.array(encode(message))
    prediction = restore_model.predict(pre)

    decision = ["Spam", "Not Spam"]
    return decision[int(prediction[0][0])]

model = build_model()
#model.summary()
#restore_model = create_restored_model()
#test_data_msg("this is a test")
