import pandas as pd
import numpy as np
columns=['city','buyer_id','seller_id','amt']
index=['a','a','b','b','c']
p=np.random.random((5,1))*500
r=np.mat([1,1,2,2,3]).T
print r.shape,p.shape
t=np.mat([1,2,3,4,5]).T
d=np.concatenate((r,t,p),1)
print d.shape
A=pd.DataFrame(d)
p=index
A.insert(0,'city',p)
A.index=range(1,6);A.columns=columns;
print A

​

