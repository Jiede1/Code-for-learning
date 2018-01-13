import random
import string
file=open("C:\\Users\\jiede\\Desktop\\2.txt",'r+')   #或者加r，防止转义
print (random.randint(12, 20))
'''for i in range(1000):
	n=random.randint(3,6)
	ip='.'.join([str(random.randint(100,200))]*3)
	string='www.'+' '.join(random.sample(['a','b','c','d','e','f','g','h','i','j','a'],n)).replace(" ","")\
	+'.com'
	file.write(ip+'  '+string+'\n')
	#print(ip+'  '+string)
for i in range(100):
	ip='.'.join([str(random.randint(100,200))]*3)
	string='www.'+' '.join(random.sample(['baidu','baid','bai','dd','tieba','wwdf',' ','u'],2)).replace(" ","")\
	+'.com'
	file.write(ip+'  '+string+'\n')
	print(ip+'  '+string)
file.close()'''
A=dict()
count=0
for line in file.readlines():
	line=line.strip().split('  ')
	if line[0]!='':
		try:
			if 'baidu' in line[1]:
				if line[0] not in A.keys():
					A[line[0]]=1
				else:
					A[line[0]]+=1
			count+=1
		except IndexError as e:
			print(e)
			break
		finally:
			print('run one line over')
print(A)
print(count)
file.close()
			
		