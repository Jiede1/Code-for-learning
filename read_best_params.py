def read_BestParams(path):
    import os
    frDir=os.listdir(path)
    #print(fr)
    for fr in frDir:
        fopen=open(path+'/'+fr,'r')
        for line in fopen:
            if 'Parameters:' in line:
                ff=line[13:-2]
                print(ff)

#read_BestParams("F:\params")


def to_dat(input):
    import pandas as pd
    input_pd=pd.DataFrame(input)
    input_pd.to_pickle("F:/BestParams.dat")
def read_dat(filename):
    import pandas as pd
    fr=pd.read_pickle(filename)
    return fr
fr=read_dat("F:/BestParams.dat")
print(read_dat("F:/BestParams.dat"))