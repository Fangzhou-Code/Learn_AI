import numpy as np

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print(scores)
scores = soft_max(scores)
print(scores)
out = scores @ Value
print(out)
