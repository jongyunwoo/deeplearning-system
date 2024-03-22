import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SSL 검증 비활성화
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#데이터 불러오기
data = tf.keras.datasets.cifar100.load_data(label_mode='fine')

#train data, validation data, test data 나누기
(train_input, train_output), (test_input, test_output) = data #train 50000개, test 10000개
train_input, validation_input, train_output, validation_output = train_test_split(train_input, train_output, test_size = 0.2, random_state=42)

#train, validation, test data 정보 출력
print(len(train_input), len(train_output))
print(len(validation_input), len(validation_output))
print(len(test_input), len(test_output))
print(train_input.shape, train_output.shape)
print(validation_input.shape, validation_output.shape)
print(test_input.shape, test_output.shape)


#데이터 전처리

# 이미지 크기 조절 및 정규화
def preprocessing(image):
    image = tf.image.resize(image, (100, 100))
    image = tf.cast(image, tf.float32)/255.0
    return image

train_scaled = preprocessing(train_input)
validation_scaled = preprocessing(validation_input)
test_scaled = preprocessing(test_input)


#pretrain된 모델 가져오기
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(input_shape=(100, 100, 3), include_top=False, weights='imagenet')

#모델 구성
inputs = tf.keras.layers.Input(shape=(100, 100, 3)) #입력 이미지 크기
X = base_model(inputs, training = False) # training = False --> 모델 추론 모드 : 새로운 데이터를 얘측할 때는 False로 설정
X = tf.keras.layers.GlobalAvgPool2D()(X) # image데이터를 처리하기 위해 GlobalAvgPooling2D 사용
outputs = tf.keras.layers.Dense(100, activation='softmax')(X) # output 클래스가 100개이기 때문에 마지막 출력 100으로 설정, activation function은 softmax로 설정

#모델 셍성
model = tf.keras.Model(inputs, outputs)

#optimizer 설정
model = tf.keras.models.experimental.SharpnessAwareMinimization(model, rho = 0.05)

#EarlyStopping 설정
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

#모델 저장
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('EfficientNet with SAM.h5')

#model compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model 훈련
history = model.fit(train_scaled, train_output,  epochs=100, batch_size = 100, validation_data=(validation_scaled, validation_output), callbacks=[callback, checkpoint_cb], verbose=2)