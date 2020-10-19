import tensorflow as tf
import numpy as np
from tensorflow import keras
from googletrans import Translator

imdb = keras.datasets.imdb

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0 #padding
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

model = keras.models.load_model('first Model.h5')

# print(model.summary())


test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#시작
while True :
    words = input("please write your review\n")
    # words = "dc영화다운 개똥같은영화"
    trans = Translator()
    result = trans.translate(words, dest='en')
    words = result.text
    words = words.replace('.','')
    words = words.split(' ')
    # print(words)

    e_words = list()
    e_words.append(1)
    for word in words:
        tmp = word_index.get(word.lower())

        if tmp is not None and tmp >= 10000:
            e_words.append(3)
        elif tmp is None:
            e_words.append(2)
        else:
            e_words.append(tmp)

    while len(e_words) < 256:
        e_words.append(0)

    e_words = np.array(e_words)
    test_data_test = np.array([e_words, ])
    res = model.predict_classes(test_data_test)
    # print(res[0])
    # print(type( res[0][0]))

    if  res[0][0] < 1:
        print("부정적 리뷰입니다.")
    else :
        print("긍정적인 리뷰입니다.")

    # print(model.predict_classes(test_data_test))인




# print(test_data[0])

# print(type(test_data))
# print(type(e_words))


# print(e_words)

# print(type(e_words))
# print(type(test_data[0]))
# print(len(e_words))
# print(len(test_data[0]))
#
# print(e_words)
# print(test_data[0])


# e_words = keras.preprocessing.sequence.pad_sequences(test_data_test,
#                                                        value=word_index["<PAD>"],
#                                                        padding='post',
#                                                        maxlen=256)


#print(test_data[0])