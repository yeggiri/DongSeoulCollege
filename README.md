# DongSeoulCollege
동서울대학교 코드 모음집

#인공지능개론 (이성진)
1. mnist
   
import tensorflow as tf
import tensorflow.keras.datasets as ds
import pickle

from PIL import Image
from test import softmax 
from test import sigmoid
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = ds.mnist.load_data()
    x_train_flatten = x_train.reshape(-1, 28*28) / 255.0
    x_test_flatten = x_test.reshape(-1, 28*28) / 255.0 
    return x_test_flatten, t_test

def init_network():          # mnist 가중치 불러오기 - github에서 따로 다운받아야 함
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

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) #확률이 가장 높은 원소의 인덱스 획득
    if p == t[i]:
        accuracy_cnt += 1
        
print("정확도 : " + str(float(accuracy_cnt)/ len(x)))
