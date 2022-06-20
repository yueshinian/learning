import torch
x= torch.arange(4.0)
x.requires_grad_(True)
y = 2*torch.dot(x,x)
y.backward()    #反向求导
print(x.grad)
x.grad.zero_() #求导会累积，需要清楚
y = x.sum()
y.backward()
print(x.grad)
x.grad.zero_()
