# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:36:30 2021

@author: neera
"""

from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
from sklearn import metrics
import pandas as pd

from src.indexing.models import BaseModel

import matplotlib.pyplot as plt


class LisaBaseModel():
    def __init__(self, degree) -> None:
        self.pageCount  = degree
        self.denseArray = np.zeros((degree, 3))
        self.nuofKeys = 0
        self.keysPerPage = 0
        self.train_array =0
        self.name = 'LisaBaseLine'
        #print('In LisaBase Model pageCount = %d' %( self.pageCount ))
    
    '''
        Calculate mapping function value for all elements in the array
        We are using mapping function as key.x1+key.x2
    '''
    def mapping_function(self):
            for i in range(0,self.train_array.shape[0]):
                self.train_array[i, 3] =   self.train_array[i][0]+self.train_array [i][1]
    
    def plot_function(self):
        plt.figure(figsize=(20,1000))
        plt.plot(self.train_array[:,3], self.train_array[:,2])
        plt.xlabel('Mapped Value')
        plt.ylabel('Position Index')
        plt.show()
        return
    
    def init_dense_array(self):
        
        self.nuofKeys = self.train_array.shape[0]
        self.keysPerPage =  self.nuofKeys // self.pageCount
        self.denseArray = np.zeros((self.pageCount, 3))
        for i in range(self.pageCount):   
            self.denseArray[i][0] = self.train_array[i*self.keysPerPage,3]
            self.denseArray[i][1] = self.train_array[((i+1)*self.keysPerPage)-1,3]
            self.denseArray[i][2] = i
            
    def search_page_index(self, x):
        low = 0
        high = self.pageCount - 1
        mid = 0
        #print('searching for %d' %(x))
        while low <= high:
        
            mid = (high + low) // 2
            #print('mid is %d' %(mid))
            # If x is greater, ignore left half
            if self.denseArray[mid][1] < x:
                low = mid + 1
 
            # If x is smaller, ignore right half
            elif self.denseArray[mid][0] > x:
                high = mid - 1
 
            # means x is present at mid
            else:
                #print('\n returning page %d' %(mid))
                return mid
    
        # If we reach here, then the element was not present
        #print('\n returning page %d' %(-1))
        return -1
 
        
        
    def predict(self, query_point):     
        #print(query_point)
        #start_time = timer()
        mapped_val = query_point[0]+query_point[1]
        '''
        found = False
        for i in range(self.pageCount):
            if ((mapped_val >= self.denseArray[i][0]) and (mapped_val <= self.denseArray[i][1])):
                found = True
                break
        
        if (found == False):
            print('\n\n\nPoint Not Found_1\n\n')
            return 0

        else:
        '''
        i = self.search_page_index(mapped_val)
        if (i == -1):
            print('\n\n\nPoint Not Found_1\n\n')
            return 0

        else:
            #print('Point found in page %d'%(i))
            page_lower = i*self.keysPerPage
            for j in range(page_lower,page_lower+self.keysPerPage):
                if ((query_point[0] == self.train_array[j][0]) and (query_point[1] == self.train_array[j][1])):
                    #print( 'value found in location %d '%(in_data_arr[j][2]))
                    #print('Time taken %f'%(timer()-start_time))
                    return self.train_array[j][2]
            print('\n\n\nPoint Not Found_2\n\n')
            
            return 0
        
    
    def range_query(self):
        '''
        i1 =0
i2 = 0
query_l = (50255,50255)
query_u = (2146566186, 2146566186)
qmap_l = query_l[0]+query_l[1]
qmap_u = query_u[0]+query_u[1]
print('qmap_l = %d qmap_u = %d'%(qmap_l, qmap_u))
for i in range(PageCount):
     if ((qmap_l >= denseArray[i][0]) and (qmap_l <= denseArray[i][1])):
            i1 = i
            lowerBound = True
            break
if(lowerBound == False):
    print('Query Rectangle Outside the range')
else:
    print('lowerbound is equal to %d'%(i))
    i = i+1
    while(i < PageCount):
        if ((qmap_u < denseArray[i][0])):
            print(denseArray[i][0])
            break
        i= i+1
    i2 = i-1
   
        '''
   

    def train(self, x_train, y_train, x_test, y_test):
        
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        #print(y_train)
        
        np.set_printoptions(threshold=1000)
        start_time = timer()
        self.train_array = np.hstack((x_train, y_train.reshape(-1,1),
                    np.zeros((x_train.shape[0], 1), dtype=x_train.dtype)))
        self.train_array  = self.train_array .astype('float64') 
        self.mapping_function( )
       
                
        # Sort the input data array with mapped values
        self.train_array  = self.train_array [self.train_array [:,3].argsort()]
        #self.plot_function(in_data_arr)
     
        #Init dense array with sorted mapped values(Store first and last key per page)
        self.init_dense_array()   
        #print(self.denseArray)
        end_time = timer()
        print('/n build time %f' %(end_time-start_time))
        test_data_size = x_test.shape[0]
        pred_y = []
        #for i in range(20):
        for i in range(test_data_size):
            pred_y.append(self.predict(x_test[i]))
        
        pred_y = np.array(pred_y)
        mse = metrics.mean_absolute_error(y_test, pred_y)
        return mse, end_time - start_time
        
        