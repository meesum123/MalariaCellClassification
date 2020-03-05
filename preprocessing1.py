# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 01:30:46 2019

@author: meesum
"""

import os

list = os.listdir('D:/MalariaCellClassification/cell_images/training_set');
olddirectory = 'D:/MalariaCellClassification/cell_images/training_set/'
number_files = len(list)
print(number_files)
print(list[0])
directory = 'D:/MalariaCellClassification/cell_images/test_set/'
name = list[0]
print(name)

for x in list :
    print(x)
    directory = directory + x  
    '''print(directory)'''
    os.mkdir(directory)
    olddirectory = olddirectory + x
    '''print(olddirectory)'''
    newlist = os.listdir(olddirectory)
    for y in range(0,3000):
        os.rename(olddirectory + '/' + newlist[y],directory + '/'  + newlist[y] )
    
    olddirectory = 'D:/MalariaCellClassification/cell_images/training_set/'
    directory = 'D:/MalariaCellClassification/cell_images/test_set/'    
    
