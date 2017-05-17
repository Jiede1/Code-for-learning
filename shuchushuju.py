#coding:utf-8
import random
from numpy import *
import os
print(os.getcwd())
os.chdir('D:\\')
f=open('f.txt','w')
def random_int_list(start, stop, length):#产生start到stop范围的随机数组
    start, stop = (int(start), in (stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

list1=random_int_list(0,500,2000)
list2=random_int_list(500,1000,2000)

for i in range(1,2000):
	f.write(str(list1[i]))
	f.write('\t')
	f.write(str(list2[i]))
	f.write('\n')
f.close()
f=open('f.txt','r')
print size(f)
for line in f:
	print type(line)

 