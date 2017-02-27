import os
from os import path
from random import random
import numpy as np


file_path = input("give the output path")
if path.exists(file_path):
    os.remove(file_path)

if not path.isdir(path.dirname(file_path)):
    os.makedirs(path.dirname(file_path))
file = open(file_path, 'w')

x = np.random.random_sample(40) * 100
for i in range(0, 40):
    x[i] = int(x[i])
    y = 2 * x[i] + 3 + (random()*10 - 5)
    file.write(str(x[i])+','+str(y)+'\n')
