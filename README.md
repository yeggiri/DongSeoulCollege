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




#로봇팔 (심원희)
1. MiniMax 알고리즘
import math

# 보드 상태를 나타내는 클래스
class Board:
    def __init__(self):
        self.cells = [' '] * 9
    
    # 보드를 출력하는 함수
    def display(self):
        for i in range(0, 9, 3):
            print('|'.join(self.cells[i:i+3]))
    
    # 비어 있는 셀을 찾는 함수
    def empty_cells(self):
        return [i for i, cell in enumerate(self.cells) if cell == ' ']
    
    # 게임 종료 여부를 확인하는 함수
    def game_over(self):
        # 가로, 세로, 대각선에 세 개의 동일한 기호가 있는지 확인
        for combo in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.cells[combo[0]] == self.cells[combo[1]] == self.cells[combo[2]] != ' ':
                return True
        # 모든 셀이 채워졌는지 확인
        if ' ' not in self.cells:
            return True
        return False
    
    # AI가 최적의 수를 찾는 MiniMax 알고리즘
    def minimax(self, depth, player):
        if player == 'X':
            best = [-1, -math.inf]
        else:
            best = [-1, math.inf]
        
        if depth == 0 or self.game_over():
            score = self.evaluate()
            return [-1, score]
        
        for cell in self.empty_cells():
            self.cells[cell] = player
            if player == 'X':
                score = self.minimax(depth - 1, 'O')
            else:
                score = self.minimax(depth - 1, 'X')
            self.cells[cell] = ' '
            score[0] = cell
            
            if player == 'X':
                if score[1] > best[1]:
                    best = score
            else:
                if score[1] < best[1]:
                    best = score
        
        return best
    
    # 보드 평가 함수
    def evaluate(self):
        # 게임이 끝났을 때 보드의 상태를 평가하여 스코어를 반환
        if self.game_over():
            for combo in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
                if self.cells[combo[0]] == self.cells[combo[1]] == self.cells[combo[2]] != ' ':
                    if self.cells[combo[0]] == 'X':
                        return 10
                    else:
                        return -10
            return 0
        else:
            return 0

# 게임 플레이 함수
def play():
    board = Board()
    board.display()
    
    while not board.game_over():
        # 플레이어의 차례
        move = int(input("어디에 둘까요? (0-8): "))
        if move not in board.empty_cells():
            print("잘못된 위치입니다. 다시 선택하세요.")
            continue
        board.cells[move] = 'X'
        board.display()
        
        if board.game_over():
            break
        
        # AI의 차례
        print("AI의 차례입니다...")
        best_move = board.minimax(3, 'O')[0]
        board.cells[best_move] = 'O'
        board.display()
    
    if board.evaluate() == 10:
        print("당신이 이겼습니다!")
    elif board.evaluate() == -10:
        print("AI가 이겼습니다!")
    else:
        print("무승부입니다.")

# 게임 실행
if __name__ == "__main__":
    play()

