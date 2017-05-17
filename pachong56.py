#!/usr/bin/python
# -*- coding: utf-8 -*-

import urllib

import os,datetime,string

import sys

from bs4 import BeautifulSoup

import re
reload(sys)

sys.setdefaultencoding('utf-8')

__BASEURL__ = 'http://bj.58.com/'

__INITURL__ = "http://bj.58.com/shoujiweixiu/"

html=urllib.urlopen(__INITURL__).read()
pattern=re.compile('<div class="tdiv">.*?<a href=\'(.*?)\'>',re.S)
data=re.findall(pattern,html)
print data[0:6],'\n'