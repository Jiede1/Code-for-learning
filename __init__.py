import re
mysent='1 3 5 2 6,75 __ '
regEx=re.compile('\\W*')
list=regEx.split(mysent)
print list
print mysent.split()