#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs

def main():
    fin = open('D:/libsvm-3.21/dataset/mnist')
    fout = codecs.open('mnist.txt','w','utf-8')
    for line in fin:
        line = line.strip().split(' ')
        label = line[0]
        fout.write(label+'\n')
    fout.close()

  
if __name__=="__main__":
    main()
