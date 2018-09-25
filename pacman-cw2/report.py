# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:22:37 2018

@author: HP
"""

import glob
filenames=glob.glob('./*.txt')
numFiles = len(filenames)


scores=[]
for i in range(numFiles):
    with open(filenames[i],'r') as f:
        score = f.readlines()
        
    