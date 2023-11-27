'''
代码参考：https://zhuanlan.zhihu.com/p/483594654

新增感知机算法原始形式代码，例题2.1
'''

import numpy as np

class perceptron:
    def __init__(self, w, b, lr):
        self.w = w
        self.b = b
        self.lr = lr

    def fit(self, x_train, y_train):
        i = 0
        iter = 1
        while i < x_train.shape[0]:
            x = x_train[i]
            y = y_train[i]
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w = self.w + self.lr * np.dot(x, y)
                self.b = self.b + self.lr * y
                print(f'iter={iter}, x=x[{i+1}], w={self.w}, b={self.b}')
                i = 0
                iter += 1
            else:
                i += 1

    def fit2(self, x_train, y_train):
        i = 0
        iter = 1
        while i < x_train.shape[0]:
            if  y_train[i] * (np.dot(self.w, x_train[i]) + self.b) <= 0:
                self.w = self.w + self.lr * np.dot(x_train[i], y_train[i])
                self.b = self.b + self.lr * y_train[i]
                print(f'iter={iter}, x=x[{i+1}], w={self.w}, b={self.b}')
                i = 0
                iter += 1
            else:
                i += 1

if __name__=='__main__':
    x_train = np.array([[3,3],[4,3],[1,1]])
    y_train = np.array([1,1,-1])
    w0 = np.zeros(x_train.shape[1])
    b0 = 0
    lr = 1
    perceptron = perceptron(w=w0, b=b0, lr=lr)
    perceptron.fit2(x_train=x_train, y_train=y_train)
    print(f'final w={perceptron.w}, b={perceptron.b}')