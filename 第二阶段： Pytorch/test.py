# 张量的创建
import numpy as np
import torch
import time



#===== 1. 通过torch.tensor()来创建张量 =====
# flag = True
flag = False
if flag:
    start_time = time.time()
    arr = np.ones((3,3))
    print("ndarray数据类型：", arr.dtype)

    # macbook m2运行pytorch在gpu上使用mps，GPU不支持float64
    # gpu运行小数据比cpu慢，数据要从gpu传到cpu
    t = torch.tensor(arr,device='mps:0',dtype=torch.float32)
    # 运行在CPU上
    # t = torch.tensor(arr)

    end_time = time.time()
    print(t)
    print("运行时间：",end_time-start_time)
    print("修改前numpy array:", arr)
    print("修改前tensor:", t)
    t[0, 0] = 0
    print("修改后numpy array:", arr)
    print("修改后tensor:", t)

#===== 2.  创建tensor，共享内存 =====
# flag = True
flag = False
if flag:
    arr = np.ones((3, 3))
    t = torch.from_numpy(arr)
    print("修改前numpy array:", arr)
    print("修改前tensor:",t)
    t[0,0] = 0
    print("修改后numpy array:", arr)
    print("修改后tensor:",t)
    print('共享内存' if arr[0,0] == t[0,0] else '不共享内存')
    print(t.data,t.type,t.shape,t.device,t.requires_grad,t.grad,t.grad_fn,t.is_leaf)
    print(id(t.data), id(arr.data), id(t.data) == id(arr.data ))


    # print('numpy 和torch互相转换2')
    # a = np.array([1, 2, 3], dtype=np.float32)
    # b = torch.Tensor(a)
    # b[0] = 999
    # print('共享内存' if a[0] == b[0] else '不共享内存')
    # print(id(a), id(b), id(a.data) == id(b.data))




#===== 3. torch.zeros(),torch.zeros_like()创建张量 =====
# flag = True
flag = False
if flag:
    out_t = torch.tensor([3])
    t = torch.zeros((3,3), out=out_t)
    t1 = torch.zeros_like(t)
    print(t,'\n',out_t,'\n',t1)
    print(id(t),id(out_t),id(t1),  id(t)==id(out_t))

#===== 4. torch.ones(),torch.ones_like()创建张量 =====
# flag = True
flag = False
if flag:
    out_t = torch.tensor([3])
    t = torch.ones((3,3), out=out_t)
    t1 = torch.ones_like(t)
    print(t,'\n',out_t,'\n',t1)
    print(id(t),id(out_t),id(t1),  id(t)==id(out_t))


#===== 5.torch.full(),torch.full_like()创建张量 =====
# flag = True
flag = False
if flag:
    out_t = torch.tensor([0])
    t = torch.full(size=(3,3),fill_value=1,out=out_t)
    t1 = torch.full_like(input=t,fill_value=2)
    print(t, '\n', out_t, '\n', t1)
    print(id(t), id(out_t), id(t1), id(t) == id(out_t))


#===== 6. torch.normal创建正态分布张量
# flag = True
flag = False
if flag:
    mean = torch.arange(1,5,dtype=torch.float)
    std = torch.arange(1.2, 5,dtype=torch.float)
    t_normal = torch.normal(mean,std)
    print("maen:{}\nstd:{}".format(mean,std))
    print(t_normal)



# 计算图
# flag = True
flag = False
if flag:
    x = torch.tensor([2.], requires_grad=True)
    print(x)
    w = torch.tensor([1.],requires_grad=True)
    print(w)
    a = torch.add(x,w)
    a.retain_grad()
    b = torch.add(w,1)
    y = torch.mul(a,b)
    y.backward()
    print(w.is_leaf,x.is_leaf,a.is_leaf,b.is_leaf,y.is_leaf)
    print(w.grad,x.grad,a.grad,b.grad,y.grad_fn)


# nn.Linear
flag = True
# flag = False
if flag:
    from torch import  nn
    model = nn.Linear(10,5)
    input = torch.rand([20,10])
    print('input={}'.format(input))
    output = model(input)
    print('output={}'.format(output))
    # for param in model.parameters():
    #     print(param)
    print(model.weight)