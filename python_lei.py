'''class Student(object):
	def __init__(self,name,age):
		self.__name=name
		self.__age=age
	def print_in(self):
		print self.__name,self.__age
bob=Student('bob',12)
#print bob.name
bob.print_in()
bob.sex='man'
print bob.sex'''
'''import logging
def foo(s):
    n = int(s)
    return 10 / n

def bar(s):
    try:
        return foo(s) * 2
    except StandardError, e:
        print 'Error!'
        logging.exception(e)

def main():
    bar('0')

main()'''
import logging
import os
print os.getcwd()

s = '0'
n = int(s)
logging.info('n = %d' % n)
#print 10 / n 
logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%a, %d %b %Y %H:%M:%S',  
                    filename='E:\test.log',  
                    filemode='w')  
  
logging.debug('debug message')  
logging.info('info message')  
logging.warning('warning message')  
logging.error('error message')  
logging.critical('critical message') 
import os
import sys
print sys.path
print os.path
print os.path.abspath('')
print os.getcwd() 
print os.environ  

import json

class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

s = Student('Bob', 20, 88)
def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }

print(json.dumps(s, default=student2dict))
def dict2student(d):
    return Student(d['name'], d['age'], d['score'])

json_str = '{"age": 20, "score": 88, "name": "Bob"}'
print(json.loads(json_str, object_hook=dict2student))