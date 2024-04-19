#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from utils import get_one_hot_label
import math
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


def load_MNIST(dataroot,rand_seed):
    training_images_filepath = dataroot + 'train-images.idx3-ubyte'
    training_labels_filepath = dataroot + 'train-labels.idx1-ubyte'
    test_images_filepath = dataroot + 't10k-images.idx3-ubyte'
    test_labels_filepath = dataroot + 't10k-labels.idx1-ubyte'
    
    Dataloader = MnistDataloader(training_images_filepath,training_labels_filepath,test_images_filepath,test_labels_filepath)
    (x_train, y_train),(x_test, y_test) = Dataloader.load_data()
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test =  np.array(y_test)
    
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
    
    y_train = get_one_hot_label(y_train,classes=10)
    y_test = get_one_hot_label(y_test,classes=10)
    
    # transform
    x_train = ((x_train / 255) - 0.1307) / 0.3081
    x_test = ((x_test / 255) - 0.1307) / 0.3081
    
    # shuffle train/valid
    index = [i for i in range(x_train.shape[0])]
    np.random.seed(rand_seed)
    np.random.shuffle(index)
    x_val = x_train[index[0:5000], :]
    y_val = y_train[index[0:5000],:]
    x_train = x_train[index[5000:60000], :]
    y_train = y_train[index[5000:60000]]
    
    return (x_train,y_train,x_val,y_val,x_test,y_test)


    
def load_minibatch(X, Y, batch_size=64, seed=1234):
    """
    从（X，Y）中创建一个随机的mini-batch列表

    参数：
        X - 输入数据，维度为(样本的数量, 数据维度)
        Y - 对应的是X的标签
        batch_size - 每个mini-batch的样本数量
    返回：
        batch_size - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）
    """

    np.random.seed(seed)  # 指定随机种子
    m = X.shape[0]
    mini_batches = []

    # 第一步：打乱顺序
    permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[permutation,:]  # 将每一列的数据按permutation的顺序来重新排列。
    # shuffled_Y = Y[permutation,:].reshape((m, Y.shape[1]))
    shuffled_Y = Y[permutation,:]
    
    # 第二步，分割
    num_complete_minibatches = math.floor(m / batch_size)  # 把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size,:]
        mini_batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
    # 如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
    if m % batch_size != 0:
        # 获取最后剩余的部分
        mini_batch_X = shuffled_X[batch_size * num_complete_minibatches:,:]
        mini_batch_Y = shuffled_Y[batch_size * num_complete_minibatches:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


    

# (x_train,y_train,x_val,y_val,x_test,y_test) = load_MNIST(dataroot='./data/',rand_seed=1234)
# mini_batches = load_minibatch(x_train, y_train, batch_size=64, seed=1234)
# # print(len(mini_batches))
# # print(len(mini_batches[0]))
# for x,y in mini_batches:
#     print(x.shape)
#     print(np.max(x,axis=-1).shape)

# print(x_train.shape)
# print(y_train.shape)

# print(x_val.shape)
# print(y_val.shape)

# print(x_test.shape)
# print(y_test.shape)