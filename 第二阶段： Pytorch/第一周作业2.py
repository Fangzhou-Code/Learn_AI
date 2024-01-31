import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)
'''
题目：调整线性回归模型停止条件以及y=3*x+(5 + torch.randn(200，1))中的斜率，训练一个线性回归模型

思路：
1. 生成数据
2. 选择模型：这里是自己构造的线性模型
3. 选择loss
4. 选择优化器
5. 训练模型
    * 前向传播-计算loss
    * loss-反向传播
'''
# 1. 生成数据
x = torch.rand(200,1) * 10
y = 3*x+(5 + torch.randn(200,1))
best_loss = float('inf')
lr = 0.01

# 2. 生成模型
w = torch.randn(1,requires_grad=True)
b = torch.randn(1, requires_grad=True)
best_w = w
best_b = b

# 3. 训练模型
for iter in range(10000):
    # 前向传播
    y_pred = w * x + b
    # 计算loss
    loss = (0.5 * (y_pred - y)**2).mean()
    # 反向传播
    loss.backward()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_w = w
        best_b = b

    if loss.data.numpy() < 3:
        print("loss.data.numpy:",loss.data.numpy(),"loss.item:", loss.item())
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2,20,'Loss=%.4f' % loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.xlim(1.5,10)
        plt.ylim(8,40)
        plt.title("iteration:{}\nw:{} b:{}".format(iter,w.data.numpy(),b.data.numpy()))
        # plt.show()
        # plt.pause(0.5)
        if loss.data.numpy() < 0.55:
            break

    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)
print(best_loss)
print(best_w)
print(best_b)