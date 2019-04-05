import numpy as np 
import os


class Data: 
    def __init__(self):
        self.batch_index = 0
        self.images = None 
        self.lables = None 
        self.num_examples = 0 
    
    def next_batch(self, batch_size=1): 
        images = self.images[self.batch_index : self.batch_index+batch_size] 
        labels = self.labels[self.batch_index : self.batch_index+batch_size] 
        self.batch_index = self.batch_index + batch_size 
        
        return images, labels 
        
    def reset_batch(self):
        self.batch_index = 0
        

class DataSet:
    def __init__(self):
        self.train = Data() 
        self.test = Data()

def load_data(dir_path):
    data = DataSet()

    file_path = os.path.join(dir_path, "train_images.csv")
    data.train.images = np.loadtxt(file_path, delimiter=',', dtype='float32')
    data.train.num_examples = data.train.images.shape[0]

    file_path = os.path.join(dir_path, "train_labels.csv")
    data.train.labels = np.loadtxt(file_path, delimiter=',', dtype='float32')

    file_path = os.path.join(dir_path, "test_images.csv")
    data.test.images = np.loadtxt(file_path, delimiter=',', dtype='float32')
    data.test.num_examples = data.test.images.shape[0]

    file_path = os.path.join(dir_path, "test_labels.csv")
    data.test.labels = np.loadtxt(file_path, delimiter=',', dtype='float32')

    return data


if __name__ == "__main__": 
    data = load_data('./data') 
    
    batch_x, batch_y = data.train.next_batch(1) 
    print(batch_x)
    print(batch_y)




