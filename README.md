# DongSeoulCollege
동서울대학교 코드 모음집

#인공지능개론 (이성진)
1. mnist
   

import tensorflow as tf
import tensorflow.keras.datasets as ds
import pickle

from PIL import Image
from CNN import softmax 
from CNN import sigmoid
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = ds.mnist.load_data()
    x_train_flatten = x_train.reshape(-1, 28*28) / 255.0
    x_test_flatten = x_test.reshape(-1, 28*28) / 255.0 
    return x_test_flatten, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()
batch_size = 100 #배치 크기
accuracy_cnt = 0

#배치사용
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #확률이 가장 높은 원소의 인덱스 획득
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

#단일 계산
# for i in range(len(x)):
    #y = predict(network, x[i])
    #p = np.argmax(y)
    #if p == t[i]:
        #accuracy_cnt += 1       
print("정확도 : " + str(float(accuracy_cnt)/ len(x)))


2. MNIST 2

import tensorflow as tf
import tensorflow.keras.datasets as ds

import numpy as np

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) /  batch_size #return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

(x_train, t_train), (x_test, t_test) = ds.mnist.load_data()

x_train_flatten = x_train.reshape(-1, 28*28)
num_classes = 10
t_train_flatten = tf.keras.utils.to_categorical(t_train, num_classes) #원-핫 인코딩 하는 함수 사용

print(x_train_flatten.shape)
print(t_train_flatten.shape)

train_size = x_train_flatten.shape[0] #훈련 데이터 배열의 첫 번째 차원 크기(행의 수) 저장
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #훈련 데이터들 중 10장을 랜덤으로 뽑아라

x_batch = x_train_flatten[batch_mask]
t_batch = t_train_flatten[batch_mask]



# 게임 실행
if __name__ == "__main__":
    play()

