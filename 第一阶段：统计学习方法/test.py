import numpy as np


# 定义感知机模型类
class Perceptorn:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_late = 1

    # 算法主函数
    def fit(self, x_trian, y_trian):
        self.w = np.zeros(x_trian.shape[1])
        i = 0

        while i < x_trian.shape[0]:
            x = x_trian[i]
            y = y_trian[i]
            # 判断其所以点是否都没有误分类，如有更新w,b,重新判断
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w = self.w + self.l_late * np.dot(x, y)
                self.b = self.b + self.l_late * y
                i = 0
            else:
                i += 1


# 训练集
x_trian = np.array([[3, 3], [4, 3], [1, 1]])
y_trian = np.array([1, 1, -1])

# 调用
perceptorn = Perceptorn()
perceptorn.fit(x_trian, y_trian)

# 输出结果
print(perceptorn.w, perceptorn.b)
