#_*_coding:utf-8_*_
import urllib
import urllib2
import re
baseurl='http://tieba.baidu.com/p/3138733512'
def getPage(pagenum,seeLZ):
	try:
		seeLZ='?see_lz='+str(seeLZ)
		url=baseurl+seeLZ+'&pn='+str(pagenum)
		request=urllib2.Request(url)
		response=urllib2.urlopen(request)
		#print response.read()
		return response.read()
	except urllib2.URLError,e:
		if hasattr(e,'reason'):
			print u"连接百度贴吧失败,错误原因",e.reason
			return None
def gettitle():
	page=getPage(1,1)
	pattern = re.compile('<h3 class="core_title_txt.*?>(.*?)</h3>',re.S)
	result = re.search(pattern,page)
	if result:
		print result.group(1)
		return result.group(1).strip()
	else:
		return None
def getpageNum():
	page=getPage(1,1)
	pattern=re.compile('<li class="l_reply_num".*?</span>.*?<span .*?>(.*?)</span>',re.S)
	result = re.search(pattern,page)
	if result:
		print result.group(0)  #测试输出
		return result.group(1).strip()
	else:
		return None
def getContent(page):
	pattern = re.compile('<div id="post_content_.*?>(.*?)</div>',re.S)
	items = re.findall(pattern,page)
	#print items[0]
	print '\n',items[3]
	for item in items:
		print item
		print '\n'
	return items

#getpageNum()
#gettitle()		
class tool:
	removeImg=re.compile('<img.*?>| {7}|')
	removeAddr = re.compile('<a.*?>|</a>')
	#把换行的标签换为\n
	replaceLine = re.compile('<tr>|<div>|</div>|</p>')
	#将表格制表<td>替换为\t
	replaceTD= re.compile('<td>')
	#把段落开头换为\n加空两格
	replacePara = re.compile('<p.*?>')
	#将换行符或双换行符替换为\n
	replaceBR = re.compile('<br><br>|<br>')
	#将其余标签剔除
	removeExtraTag = re.compile('<.*?>')
	def replace(self,x):
		x = re.sub(self.removeImg,"",x)
		x = re.sub(self.removeAddr,"",x)
		x = re.sub(self.replaceLine,"\n",x)
		x = re.sub(self.replaceTD,"\t",x)
		x = re.sub(self.replacePara,"\n    ",x)
		x = re.sub(self.replaceBR,"\n",x)
		x = re.sub(self.removeExtraTag,"",x)
		#strip()将前后多余内容删除
		return x.strip()
page=getPage(1,1)
ab=tool()
ans=getContent(page)
#print ans[1],'\n\n'

pt=ab.replace(ans[1])
#print "pt:",pt
print '\n'," ilove you"