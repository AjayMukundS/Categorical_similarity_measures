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
def Frequency_table2(data):
 	  num_cat= []
 	  order_cat= []
 	  for col_num in range(len(data.columns)):
 	      col_name= data.columns[col_num]
 	      categories = list(data[col_name].unique())
 	      order_cat.append(categories)
 	      num_cat.append(len(categories))
 	  r = data.shape[0]
 	  s = data.shape[1]
 	  freq_table= np.zeros(shape=(max(num_cat),s))
 	  
 	  for i in range(s): 
 	      counter= 0
 	      for j in (order_cat[i]): 
 	          count= 0
 	          for k in data[data.columns[i]]:
 	              if k == j:
 	                  count= count+ 1
 	          freq_table[counter][i] = count
 	          counter= counter+1
 	  return(freq_table)

def Lin1(data):
 	  data= ordinal_encode(data)
 	  r = data.shape[0]
 	  s = data.shape[1]  
 	  
 	  freq_table= Frequency_table(data)
 	  
 	  freq_rel= freq_table/r
 	  
 	  freq_log= np.log(freq_rel)
 	  
 	  freq_log[freq_log== -math.inf]= 0
 	  
 	  agreement= []
 	  for i in range(s):
 	      agreement.append(0)
 	  
 	  lin1= np.zeros(shape=(r,r))
 	  
 	  weights= []
 	  for i in range(s):
 	      weights.append(0)    
 	  
 	  for i in range(r-1):
 	      for j in range(1+i, r):
 	          for k in range(s):
 	              c = data.iat[i,k]-1
 	              d = data.iat[j,k]-1
 	              if (data.iat[i,k] == data.iat[j,k]):
 	                  logic= freq_rel[:, k] == freq_rel[c][k]
 	                  agreement[k] = sum(logic * freq_log[:, k])
 	                  weights[k]= sum(logic * freq_log[:, k])
 	              else:                    
 	                  if freq_rel[c][k] >= freq_rel[d][k]:
 	                      logic= np.logical_and(freq_rel[:, k] >= freq_rel[d][k], freq_rel[:, k] <= freq_rel[c][k])
 	                      agreement[k]= 2* np.log(sum(logic * freq_rel[:, k]))
 	                      weights[k]= sum(logic * freq_log[:, k])
 	                  else:
 	                      logic= np.logical_and(freq_rel[:, k] >= freq_rel[c][k], freq_rel[:, k] <= freq_rel[d][k])
 	                      agreement[k]= 2* np.log(sum(logic * freq_rel[:, k]))
 	                      weights[k]= sum(logic * freq_log[:, k])
 	          if i == j:
 	              lin1[i][j]= 0
 	          else:
 	              lin1[i][j] = 1/(1/sum(weights)*(sum(agreement))) - 1
 	              lin1[j][i] = lin1[i][j]
 	  lin1[lin1== -math.inf]= lin1.max()+ 1
 	  return(lin1)
def ordinal_encode(data):
 	  encoding_data= data.copy()
 	  encoder= ce.OrdinalEncoder(encoding_data)
 	  data_encoded= encoder.fit_transform(encoding_data)
 	  return(data_encoded)
