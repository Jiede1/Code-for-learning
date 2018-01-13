import random
file=open(r"D:\公司专用\SDB文档\贵州广电项目\bak_2017_11_29.csv",'w')
nameobj=['a','b','c','d','e']
addressobj=['beijing','guanzhou','hunan','tianjin']
Date='2017-11-29'
for i in range(10000):
	name=nameobj[random.randint(0,4)]
	phone=str(random.randint(120, 20000))
	address=addressobj[random.randint(0,3)]
	money=str(random.randint(10000,30000))
	file.writelines([name,',',phone,',',address,',',money,',',Date,'\n'])
file.close()
