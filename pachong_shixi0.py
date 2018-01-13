from pyquery import PyQuery as pyq
data=''
with open(r'C:\Users\jiede\Desktop\jieshao.htm','r+') as f:
    for line in f.readlines():
        if(line.find('<meta charset="gb2312">')==0):  
            print(line)
            line='<meta charset="utf-8">'+'\n'
        data+=line
    
with open(r'C:\Users\jiede\Desktop\jieshao.htm', 'w+') as f:
    f.writelines(data)

doc=pyq(filename=r'C:\Users\jiede\Desktop\jieshao.htm')
print(doc('div.recommend_main').html())