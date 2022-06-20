import torch
a = torch.arange(20).reshape(4,5)
b=a.T
print(a,b) 
print(a.sum())
print(a.sum(axis=0))
#torch.dot(x,y )
#torch.norm(x,y)
print(torch.mm(a,b))
print(torch.norm(torch.ones(4,9)))

