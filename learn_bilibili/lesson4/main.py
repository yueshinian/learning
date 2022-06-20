from dataclasses import dataclass
import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok= True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n') #列名
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 178100\n') #数据
    f.write('4, NA, 106000\n')

data = pd.read_csv(data_file)
print(data)

##处理缺失数据，插值和删除
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) #均值填充
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x.shape) 
print(y)

