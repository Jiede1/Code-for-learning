#coding:utf-8
import urllib
def callbackfunc(blocknum, blocksize, totalsize):
    '''回调函数
    @blocknum: 已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
    '''
    percent = 100.0 * blocknum * blocksize / totalsize
    if percent > 100:
        percent = 100
    print "%.2f%%"% percent
url = 'http://www.sina.com.cn'
local = 'C:\\desktop\sina.html'
urllib.urlretrieve(url, local, callbackfunc)
