#coding:utf-8
import urllib
import re
#获取整个页面的信息
def getHTML(url):
    page=urllib.urlopen(url)
    html=page.read()
    return html
#获取页面里想要的数据
def getImage(html):
    res=r'src="(.*?\.jpg)" pic_ext'
    pic=re.compile(res)
    imglist=re.findall(pic,html)
    x=0
    for imgurl in imglist:
        print imgurl
        load=r'C:\desktop\%d.jpg'%x
        print load
        urllib.urlretrieve(imgurl,load)
        x+=1
    return imglist
#将数据保存到本地

html=getHTML('http://tieba.baidu.com/p/2460150866')
imglist=getImage(html)

print imglist
#print html