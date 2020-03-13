import pandas as pd
import numpy as np
import category_encoders as ce
import math

def Frequency_table(data):
 	  num_cat= []
 	  for col_num in range(len(data.columns)):
 	      col_name= data.columns[col_num]
 	      categories = list(data[col_name].unique())
 	      num_cat.append(len(categories))
 	  r = data.shape[0]
 	  s = data.shape[1] 
 	  freq_table= np.zeros(shape=(max(num_cat),s))
 	  
 	  for i in range(s): 
 	      for j in range(num_cat[i]): 
 	          count= []
 	          for num in range(0, r):
 	              count.append(0)
 	          for k in range(0,r):
 	              if (data.iat[k,i] -1== j): 
 	                  count[k] = 1
 	              else:
 	                  count[k] = 0
 	          freq_table[j][i] = sum(count)
 	  return(freq_table)
def VM(data):
 	  data= ordinal_encode(data)
 	  r = data.shape[0]
 	  s = data.shape[1]  
 	  
 	  num_cat= []
 	  
 	  for col_num in range(len(data.columns)):
 	      col_name= data.columns[col_num]
 	      categories = list(data[col_name].unique())
 	      num_cat.append(len(categories))
 	  
 	  freq_table= Frequency_table(data)    
 	  freq_rel= freq_table/r    
 	  freq_rel2= freq_rel**2
 	  
 	  sum_freq_rel2= []
 	  gini= []
 	  norm_gini= []
 	  for i in range(s):
 	      sum_freq_rel2.append(freq_rel2[:, i].sum())
 	      gini.append(1- sum_freq_rel2[i])
 	      norm_gini.append(gini[i]*num_cat[i]/(num_cat[i] - 1))
 	  
 	  norm_gini= np.nan_to_num(norm_gini)
 	  
 	  agreement= []
 	  for i in range(s):
 	      agreement.append(0)
 	  
 	  vm= np.zeros(shape=(r,r))
 	  
 	  for i in range(r-1):
 	      for j in range(1+i, r):
 	          for k in range(s):
 	              if (data.iat[i,k] == data.iat[j,k]):
 	                  agreement[k] = norm_gini[k]
 	              else:
 	                  agreement[k] = 0
 	          if i == j:
 	              vm[i][j]= 0
 	          else:
 	              vm[i][j] = 1-1/s*(sum(agreement))
 	              vm[j][i] = vm[i][j]
 	  return(vm)
def ordinal_encode(data):
 	  encoding_data= data.copy()
 	  encoder= ce.OrdinalEncoder(encoding_data)
 	  data_encoded= encoder.fit_transform(encoding_data)
 	  return(data_encoded)
