import os
import numpy as np
import random

def is_csv(file_name) :             #파일명이 csv파일인지 검사
    _, ext= os.path.splitext(file_name)
    return ext=='.csv'

def get_csv_list(dir_path):         #리스트에서 csv 파일만 추출
    return [ file for file in os.listdir(dir_path) if is_csv(file)]

def read_data(file_path) :          # csv파일에서 데이터 셋작성
    data = np.loadtxt(file_path, delimiter = ",", dtype='float32')
    np.random.shuffle(data)

    total = data.shape[0]
    base = int(total *0.8)
    train = data[:base]     #80% 훈련데이터
    test = data[base:]      #20% 테스트데이터

    return train, test

#주어진 크기의 배열에 one hot 표현
def one_hot(rows, cols, index) :
    arr = np.zeros([rows, cols], dtype = 'float32')
    arr[:, index] =1
    return arr

#훈련 데이터와 테스트 데이터 목록
train_datas = [] 
test_datas = []

target_dir = "C:/Users/student/Face_Recog/face_image_processing/" 
list = os.listdir(target_dir) 
for dir in list : 
    work_dir = os.path.join(target_dir,dir) 
    for csv in get_csv_list(work_dir) : 
        index, _ = os.path.splitext(csv) # 파일명이 인덱스 
        csv = os.path.join(work_dir, csv) 
        
        train_images, test_images = read_data(csv)
        train_labels = one_hot(train_images.shape[0], 22, int(index)) 
        test_labels = one_hot(test_images.shape[0], 22, int(index)) 
        
        for item in zip(train_images, train_labels): 
            train_datas.append(item) 
            
        for item in zip(test_images, test_labels):
            test_datas.append(item)

#셔플            
random.shuffle(train_datas)
random.shuffle(test_datas)

train_images = [] 
train_labels = [] 
for image, label in train_datas: 
    train_images.append(image)
    train_labels.append(label) 


test_images = [] 
test_labels = [] 
for image, label in test_datas:
    test_images.append(image) 
    test_labels.append(label) 

target_dir = "C:/Users/student/Face_Recog/data/" 
np.savetxt(target_dir + 'train_images.csv', train_images,
    fmt='%.6f', delimiter=',') 
np.savetxt(target_dir + 'train_labels.csv', train_labels,
    fmt='%.6f', delimiter=',') 
np.savetxt(target_dir + 'test_images.csv', test_images,
     fmt='%.6f', delimiter=',')
np.savetxt(target_dir + 'test_labels.csv', test_labels,
     fmt='%.6f', delimiter=',')