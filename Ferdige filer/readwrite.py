import codecs
import numpy as np

# This script read and writes from files

def write_data(data,filename): # The function writes "data" to the file with filename "filename"
    f = codecs.open(filename,"w","utf-8")
    for i in range(len(data[0])):
        f.write(str(data[0,i])+"\t"+str(data[1,i])+"\n")
    f.close()
    
def read_data(filename): # The function reads from the file with filename "filename"
    f = codecs.open(filename,"r")
    lines = f.readlines()
    data = np.zeros([2,len(lines)-1])
    for i in range(len(lines)-1):
        temp = lines[i].strip("\n").split("\t")
        data[0,i] = temp[0]
        data[1,i] = temp[1]
    f.close()
    return data

