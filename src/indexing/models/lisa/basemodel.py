# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:36:30 2021

@author: neera
"""
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('')
import src.indexing.utilities.metrics as metrics


class LisaBaseModel():
    def __init__(self, degree) -> None:
        # PageCount : Number of pages into which keyspace is divided
        self.pageCount = degree
        # denseArray = Array to store Page Addresses
        self.denseArray = np.zeros((degree, 3))
        # nuofKeys = Total Nu of Keys in Database
        self.nuofKeys = 0
        # keysPerPage = Nu of keys per page.
        self.keysPerPage = 0
        # Data Structure to hold keys-value pairs
        self.train_array = 0
        self.name = 'Lisa Baseline'
        #Book Keeping code
        self.page_size = 1
        self.debugPrint = False


    '''
        Apply mapping function(key.x1+key.x2) to keys database.
        Parameters
        ----------
        self.train_array:
             Data structure containing key-value pairs .
         
        Returns
        -------
        self.train_array:
             Update the data structure with mapped value for each key 
    '''

    def mapping_function(self):
        for i in range(0, self.train_array.shape[0]):
            self.train_array[
                i, 3] = self.train_array[i][0] + self.train_array[i][1]

    '''
        Plot mapped values against position indexes.
       
    '''

    def plot_function(self):
        plt.figure(figsize=(20, 1000))
        plt.plot(self.train_array[:, 3], self.train_array[:, 2])
        plt.xlabel('Mapped Value')
        plt.ylabel('Position Index')
        plt.show()
        return

    '''
        Initialize model parameters based on size of training data
        
    '''

    def init_dense_array(self):

        # nuofKeys will be equal to nu of data points in the training database.
        self.nuofKeys = self.train_array.shape[0]
        self.denseArray = np.zeros((self.pageCount, 3))
        # Divide the keys space into equal length intervals.
        self.keysPerPage = self.nuofKeys // self.pageCount
        # Last page may have less number of keys than keysPerPage
        if (self.nuofKeys > self.keysPerPage * self.pageCount):
            self.keysPerPage = self.keysPerPage + 1

        if (((self.keysPerPage * self.pageCount) - self.nuofKeys) >=
                self.keysPerPage):
            print(
                'Invalid configuration, Nu of keys per page needs to be greater than page count'
            )
            return -1
        # Store mapped value of first and last key of each page in denseArray
        # These values will be used to decide the page for query point during query search
        for i in range(self.pageCount - 1):
            self.denseArray[i][0] = self.train_array[i * self.keysPerPage, 3]
            self.denseArray[i][1] = self.train_array[(
                (i + 1) * self.keysPerPage) - 1, 3]
            self.denseArray[i][2] = i

        # Last page may not be full
        i = self.pageCount - 1
        #Store mapped value boundries
        self.denseArray[i][0] = self.train_array[i * self.keysPerPage, 3]
        self.denseArray[i][1] = self.train_array[self.nuofKeys - 1, 3]
        self.denseArray[i][2] = i
        return 0

    '''
       Perform binary search based on query point mapped value to find the page address
       containign the key
       Parameters
        ----------
        x : Integer 
            Mapped value of the query point
                
        Returns
        -------
        mid: Integer
           Returns the page address or -1
        
    '''

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

    '''
       Perform binary search for the query point in a page based on mapped value
       Parameters
        ----------
        x : Integer 
            Mapped value of the query point
        page_lower : Interger
            Offset of the page containg the mapped key        
        Returns
        -------
        mid: Integer
           Returns the index which matches the query point mapped value or -1 
        
    '''

    def key_binary_search(self, x, page_lower):
        low = page_lower
        # Last page may contain less nu of keys than self.keysPerPage
        if (page_lower == (self.keysPerPage * (self.pageCount - 1))):
            # Last page
            high = self.nuofKeys - 1
        else:
            high = page_lower + self.keysPerPage - 1
        mid = 0
        #print('searching for %d' %(x))
        while low <= high:

            mid = (high + low) // 2
            #print('mid is %d' %(mid))
            # If x is greater, ignore left half
            if self.train_array[mid][3] < x:
                low = mid + 1

                # If x is smaller, ignore right half
            elif self.train_array[mid][3] > x:
                high = mid - 1

            # means x is present at mid
            else:
                #print('\n returning index %d' %(mid))
                return mid

        # If we reach here, then the element was not present
        #print('\n returning page %d' %(-1))
        return -1

    '''
       Return keys belonging to range query from cells belonging to cell list
       Parameters
        ----------
        query_l : tuple
            Range Query lower coordinate
        
        query_u  : tuple
            Range Query upper coordinate
            
        cellList : List
            List contaning cells ids which are identified as part of 
        query
                     
        Returns
        -------
        keylist :  npArray
            Array of key/value pairs fetched by range query
               
                           
    '''
    def getKeysInRangeQuery(self, lowerPage, upperPage,query_l, query_u):
        keyList = []
        pageIdx = lowerPage
        while(pageIdx <=upperPage):
            pageStart = pageIdx * self.keysPerPage
            # Last page may contain less nu of keys than self.keysPerPage
            if (pageStart == (self.keysPerPage * (self.pageCount - 1))):
                # Last page
                pageEnd = self.nuofKeys
            else:
                pageEnd = pageStart + self.keysPerPage
           
            
            for j in range(pageStart, pageEnd):
                 if(self.train_array[j, 0] >= query_l[0] and self.train_array[j, 0] <= query_u[0] )and \
                         (self.train_array[j, 1] >= query_l[1] and self.train_array[j, 1] <= query_u[1] ):
                        keyList.append(self.train_array[j, 0:3])
            pageIdx += 1
     
        return np.array(keyList)
    
    '''
       Decompose range query into a union of smaller query rectangles each 
       belong to one and only one cell. 
        
        Parameters
        ----------
        query_l : tuple
            Range Query lower coordinate
        
        query_u  : tuple
            Range Query upper coordinate
                     
        Returns
        -------
        cell_list :  union of smaller query rectangles each 
       belong to one and only one cell.
            
    '''
    def range_query(self,query_l, query_u):
      
        if self.debugPrint:
              print('Get pages for range (%d, %d), (%d, %d)' %(query_l[0], query_l[1], query_u[0], query_u[1] )) 
        lowerPage = self.search_page_index( query_l[0]+query_l[1]) 
        upperPage = self.search_page_index( query_u[0]+query_u[1]) 
        if self.debugPrint:
            print("Pages to search from %d to %d" %(lowerPage,upperPage))
        return (lowerPage,upperPage)
    
    
    '''
        Predict range query for lisa model
                                
    '''
    def predict_range_query(self, query_l, query_u):
        (lowerPage,upperPage) =self.range_query(query_l, query_u) 
        if(lowerPage == -1) or (upperPage == -1)  :
            if self.debugPrint:
                print('range query not found')
            return -1
        else:
            neighboursKeySet = self.getKeysInRangeQuery(lowerPage, upperPage,query_l,query_u )
            return np.sort(neighboursKeySet[:, -1])
        


    '''
       Predict the position of query point in the database
       Parameters
        ----------
        Query_point: Tuple 
            2 dimensional key value
       
        Returns
        -------
        self.train_array[j][2]: Integer
           Returns the value at the query point
        
    '''

    def predict(self, query_point):
        #print(query_point)
        #start_time = timer()
        mapped_val = query_point[0] + query_point[1]
        i = self.search_page_index(mapped_val)
        if (i == -1):
            print(
                '\n\n\n Page not found query point = %d %d, mapped value = %d'
                % (query_point[0], query_point[1], mapped_val))
            return -1

        else:
            page_lower = i * self.keysPerPage
            # Last page may contain less nu of keys than self.keysPerPage
            if (page_lower == (self.keysPerPage * (self.pageCount - 1))):
                # Last page
                high = self.nuofKeys
            else:
                high = page_lower + self.keysPerPage

            for j in range(page_lower, high):
                if ((query_point[0] == self.train_array[j][0])
                        and (query_point[1] == self.train_array[j][1])):
                    #print( 'value found in location %d '%(in_data_arr[j][2]))
                    #print('Time taken %f'%(timer()-start_time))
                    self.train_array[j][2]
                    return self.train_array[j][2]

            print(
                '\n\n\n Point not found query point = %d %d, mapped value = %d'
                % (query_point[0], query_point[1], mapped_val))
            return -1

    '''
       Predict the position of query point based on mapped value instead of 
       sequential search. 
       Parameters
        ----------
        Query_point: Tuple 
            2 dimensional key value
       
        Returns
        -------
        self.train_array[j][2]: Integer
           Returns the value at the query point
        
    '''

    def predict_opt(self, query_point):

        mapped_val = query_point[0] + query_point[1]
        i = self.search_page_index(mapped_val)
        if (i == -1):
            print(
                '\n\n\nPage Not Found:search page return -1, for query point %d %d \n\n'
                % (query_point[0], query_point[1]))
            return i

        else:

            page_lower = i * self.keysPerPage
            # Find key index based on mapped value
            key_index = self.key_binary_search(mapped_val, page_lower)
            if (key_index != -1):
                # Multiple keys can have the same mapped value.

                if ((query_point[0] == self.train_array[key_index][0]) and
                    (query_point[1] == self.train_array[key_index][1])):
                    # Return value if index key value matches with query point
                    return (self.train_array[key_index][2])
                else:
                    i = 0
                    # Search in the neighbourhood of index returned by key_binary_search
                    # as multiple keys can have the same mapped value
                    while (mapped_val == self.train_array[key_index - i][3]):
                        if ((query_point[0]
                             == self.train_array[key_index - i][0])
                                and (query_point[1]
                                     == self.train_array[key_index - i][1])):
                            return (self.train_array[key_index - i][2])
                        else:
                            i = i + 1
                    i = 0
                    while (mapped_val == self.train_array[key_index + i][3]):
                        if ((query_point[0]
                             == self.train_array[key_index + i][0])
                                and (query_point[1]
                                     == self.train_array[key_index + i][1])):
                            return (self.train_array[key_index + i][2])
                        else:
                            i = i + 1
                print(
                    '\n\n\n Point not found query point = %d %d, mapped value = %d'
                    % (query_point[0], query_point[1], mapped_val))
                return -1
            else:
                print(
                    '\n\n\n Point not found query point = %d %d, mapped value = %d'
                    % (query_point[0], query_point[1], mapped_val))
                return -1

    '''
       Train the lisa Baselinemodel: Training consists of:
            a) Applying mapping function to keys values taking into account '
               cell boundaries
            b)Divining mapped values into equal length intervals. 
    
       Parameters
        ----------
        Train and test point np arrays
       
        Returns
        -------
        mse: Float
           Mean square error for eval points
           time : Time taken to build the lisaBaseline model. 
        
    '''

    def train(self, x_train, y_train, x_test, y_test):

        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        np.set_printoptions(threshold=1000)
        start_time = timer()
        self.train_array = np.hstack((x_train, y_train.reshape(-1, 1),
                                      np.zeros((x_train.shape[0], 1),
                                               dtype=x_train.dtype)))
        self.train_array = self.train_array.astype('float64')
        # Apply mapping function to 2 dimenional key values
        self.mapping_function()

        # Sort the input data array with mapped values
        self.train_array = self.train_array[self.train_array[:, 3].argsort()]
        #self.plot_function(in_data_arr)

        #Init dense array with sorted mapped values(Store first and last key per page)
        if (self.init_dense_array() == -1):
            return -1, timer() - start_time

        end_time = timer()
        print('/n build time %f' % (end_time - start_time))
        test_data_size = x_test.shape[0]
        pred_y = []
        #for i in range(20):
        print('\n In Lisabaseline.build evaluation %d data points' %
              (test_data_size))
        for i in range(test_data_size):
            pred_y.append(self.predict(x_test[i]))

        pred_y = np.array(pred_y)
        mse = metrics.mean_squared_error(y_test, pred_y)
        return mse, end_time - start_time
