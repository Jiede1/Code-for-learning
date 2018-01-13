import re
line='1\\xsd"china",1,"good","20170807"'
line_after_code = re.sub(r'[\\x1d]', '6?', line)  #[\x1d]
print(len(line_after_code),type(line_after_code))
print(line_after_code,line)
