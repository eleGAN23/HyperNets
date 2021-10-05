# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:22:06 2020

@author: Edoardo
"""

def readFile(path):
    
    with open(path, 'r') as f:
        r = f.read()
        r = r.replace('=', '+').replace('\n', '+').split('+')
        new_r = []
        for i in r:
            if i=='True':
                new_r.append('1')
            elif i == 'False':
                new_r.append(0)
            elif i !='' and '#' not in i:
                new_r.append(i)
    # print(new_r)
    return new_r
        
# readFile('TrainingArguments.txt')    