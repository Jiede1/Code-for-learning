#coding:utf-8
#文件名：meizi_page_download
import urllib2
import os
import re
#loadurl()这个函数呢，是防打开链接超时，如果超时返回空字符，则主调函数会再次调用(while语句就可以实现)，正常的话返回html代码，一个网页不算大，如果你的网络超级好，timeout可以缩短
def loadurl(url):
    try:
        conn = urllib2.urlopen(url,timeout=5)
        html = conn.read()
        return html
    except urllib2.URLError:
        return ''
    except Exception:
        print("unkown exception in conn.read()")
        return ''
#这里是图片保存的代码被调函数，timeout=5设置超时时间，一个500k不到的图片，5秒时间算长的了，超时的话，返回失败

def download(url,filename):
    try:
        conn = urllib2.urlopen(url,timeout=5)
        f = open(filename,'wb')
        f.write(conn.read())
        f.close()
        return True
    except urllib2.URLError:
        print 'load',url,'error'
        return False
    except Exception:
        print("unkown exception in conn.read()")
        return ''

#保存图片的逻辑代码块
def save_pic(url,path):
    searchname = '.*/(.*?.jpg)'
    name = re.findall(searchname,url)
    filename = path +'/'+ name[0]

    print filename + ':start' #控制台显示信息
    #下面的代码，当下载成功，break跳出就好了，如果存在，直接结束这个函数

    #定义了在下载图片时遇到错误的重试次数
    tryTimes = 3

    #当重试次数没有用完时，则尝试下载
    while tryTimes != 0:
        tryTimes -= 1
        if os.path.exists(filename):
            print filename,' exists, skip'
            return True
        elif os.path.exists(filename):
            os.mknod(filename)
        if download(url,filename):
            break

    if tryTimes != 0:
        print(filename + ": over")
    else:
        print(url + " ：Failed to download")
    #控制台显示信息

#这个函数，相当于一个中介，我只是把for循环代码提出就得到了这个函数    
def pic_list(picList,path):
    picurl = ''
    for picurl in picList:
        save_pic(picurl,path)

#图片下载的主逻辑函数，获取图片链接，然后传给pic_list()，等结果(其实也没结果，就是等退出)
def picurl(url,path):
    if os.path.exists(path):
        print path, 'exist'
    else:
        os.makedirs(path)
    html = ''
    while True:#这里和下载图片是一个道理，细看即可
        html = loadurl(url)
        if html == '':
            print 'load', url,'error'
            continue
        else:
            break
    #其实这里呢，也是后期发现的一个小bug，这个网站的前后代码有不同（目前而言发现的一处），在rePicContent1运行到后面，是匹配不到的，导致rePicList返回的结果也是空，也就造成了这个符号[0]报错，因为没有任何值，越界错误（我估计学过编程的，对这个耳熟能详吧），单线程会在这里报错并停止运行。rePicContent2其实是我解决bug的另一个匹配正则式，被我发现的页面是这个--http://www.meizitu.com/a/454.html，有兴趣的去对比看看
    rePicContent1 = '<div.*?id="picture.*?>.*?<p>(.*?)</p>'
    rePicContent2 = '<div.*?class="postContent.*?>.*?<p>(.*?)</p>'
    rePicList = '<img.*?src="(.*?)".*?>'
    #这里对re.S做个介绍，re.S是可以不添加的，加上之后，它的作用就是能忽略换行符，将两条作为一条来匹配。html代码碰上换行的概率是很高的，所以我一致采用re.S(下文有配图)
    picContent = re.findall(rePicContent1, html,re.S)
    if len(picContent) <=0:
        picContent = re.findall(rePicContent2, html,re.S)
    if len(picContent) <=0:
        print 'load false, over download this page and return'
        return False
    else:
        picList = re.findall(rePicList,picContent[0],re.S)
        pic_list(picList,path)
		
#文件名：meizi_series_nextpage
import re
import urllib2
#这个呢，是获取组图套图的代码，是下一个需要显示的代码块
#同样的，这里是加载链接防超时，和上一节一样
def loadurl(url):
    try:
        conn = urllib2.urlopen(url, timeout=5)
        html = conn.read()
        return html
    except urllib2.URLError:
        return ""
    except Exception:
        print("unkown exception in conn.read()")
        return ""

#上述代码中，最后还有一个except Exception，用于处理URLErro类无法捕捉的其他异常。感谢实验楼用户@caitao。



#下面的这个path指的是保存本地的文件路径，我在第一小节已经讲过了，还记得么？跟着代码再将一次吧
def nextpage(url,path):
    reNextLink = "<a.*?href='(.*?)'>.*?</a>"
    #获取reNextPage里的标签的全部链接
    reNextPage = '<div.*?id="wp_page_number.*?>.*?<ul>(.*?)</ul>'
    #嘿嘿，获取ul标签里面的内容，里面包含了所有我们需要的链接，找到wp_page_number就可以了
    #下面呢，目的是获取链接名，组合传入路径得到当前路径名，解释：匹配a到z字符，>=1个
    searchPathTail = '.*/([a-z]+).*?.html'
    #获取传入的链接尾巴
    searchurltail = '.*/(.*?.html)'
    #获取传入的链接头部
    searchhead = '(.*)/.*?.html'
    #分开头和尾，是因为在获取当前标签的所有页码，都不是完整的，而是尾部链接，需要用尾部和头部链接拼凑成完整的链接。头部链接呢，就是传入链接的头部，而且传入的是第一个完整链接，页面1里面又没有尾部链接，所有传入链接的尾部，也需要找出
    pathTail = re.findall(searchPathTail,url,re.S)
    urlTail = re.findall(searchurltail,url,re.S)
    urlhead = re.findall(searchhead,url,re.S)
    #从传入文件夹路径和从链接中分析出的文件名，得到当前文件夹路径，保存到path中
    path = path + '/' +pathTail[0]
    print path
    #标签页面的存储列表nextpage
    nextpageurl = []
    html = ''
    while True:
        html = loadurl(url)
        if html == '':
            print 'load', url,'error'
            continue
        else:
            break
    nextPage = re.findall(reNextPage,html,re.S)
    nextLink = re.findall(reNextLink,nextPage[0],re.S)
    nextLink.append(urlTail[0])
    #这一段呢，是将标签页码的所有尾部链接保存到nextLink中，然后下面的for循环，将完整的url链接，存储到nextpageurl中
    nextLink = sorted(list(set(nextLink)))
    for i in nextLink:
        nextpageurl.append(urlhead[0]+"/"+i)
    #将url链接和对应的文件路径传入"获取标签第n页的所有组图链接"的模板中，引号标记的，就是下一个代码块
    for i in nextpageurl:
        print i
        tag_series(i,path)

		
#文件名：meizi_series_getpage
import re
import urllib2

def loadurl(url):
    #依旧的，防超时和循环加载
    try:
        conn = urllib2.urlopen(url,timeout=5)
        html = conn.read()
        return html
    except urllib2.URLError:
        return ''
    except Exception:
        print("unkown exception in conn.read()")
        return ''

#这个函数呢，简单点就是根据套图链接和传入的路径，得到套图文件夹路径，再传给上一节的图片下载模板
def oneOfSeries(urllist,path):
    searchname = '.*/(.*?).html'
    current_path = '' 
    for url in urllist:
        try:
            name = re.findall(searchname,url,re.S)
            current_path = path + '/' + name[0]
            picurl(url,current_path)
        except urllib2.URLError:
            pass

#传入标签的第n页和文件夹路径，获取所有套图url链接，和分析出对应的文件夹路径，传给我们底层的图片下载模板（也就是上一节啦）
def tag_series(url,path):
    #这里呢，是直接匹配出套图的链接，直接，注意是直接，这里呢最好是将结果和源码对下结果，防止遗漏和多出
    reSeriesList = '<div .*?class="pic".*?>.*?<a.*?href="(.*?)".*?target.*?>'
    html = ''
    while True:
        html = loadurl(url)
        if html == '':
            print 'load', url,'error'
            continue
        else:
            break
    seriesList = re.findall(reSeriesList,html,re.S)
    if len(seriesList) ==0:
        pass
    else:
        oneOfSeries(seriesList,path)
		
		
#来看看，这个是前面说烂了的，放网络超时和循环直至打开为止
def loadurl(url):
    try:
        conn = urllib2.urlopen(url,data=None,timeout=5)
        html = conn.read()
        return html
    except Exception:
        return ''

#下面是主函数，细讲
def meizi(url,path):
    #见上面的html代码截图，对比无误
    reTagContent = '<div.*?class="tags">.*?<span>(.*?)</span>'
    reTagUrl = '<a.*?href="(.*?)".*?>'
    print 'start open meiziwang'
    html = ''
    while True:
        html = loadurl(url)
        if html == '':
            print 'load', url,'error'
            continue
        else:
            break
    tagContent = re.findall(reTagContent, html, re.S)
    taglists = re.findall(reTagUrl, tagContent[0], re.S)
    #而且，如果你仔细看，你会发现，链接又重，而且匹配、添加到列表，重复依旧在，所以啦，需要去重和排序，
    taglists = sorted(list(set(taglists)))
    for url in taglists:
        nextpage(url,path)

meizi('http://www.meizitu.com','C:\\Desktop')
print 'Spider Stop'