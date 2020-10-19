import tensorflow as tf
from tensorflow import keras
import numpy as np

#1. IMDB 데이터셋 다운로드
#2. 데이터 탐색(어떤 구조인지 파악)
#3. 정수를 단어로 다시 변환하기
#4. 데이터 준비단계(신경망에 주입하기 위해 텐서의 크기를 동일하게 맞춤 -> padding 함수를 통해)
#5. 모델 계층구조 잡



imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()


# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
# print(word_index.items())
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0 #padding
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#모델 구조 생성
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#모델 옵티마이저와 손실함수 구성하
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 검증 세트 생성 ->훈련 데이터만을 사용하여 모델을 개발하고 튜닝하는 것이 목표이기 때문에 테스트 데이터 사용 X
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#학습 시작
print("Learning Start")
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

print("Learning Ended")

#모델 평
results = model.evaluate(test_data,  test_labels, verbose=2)

# print(results)
# print(model.predict(test_data[0]))
print(model.predict_classes(test_data[:2]))

# type(test_data[:2])

#print(test_data[0])
#model.save('first Model.h5')