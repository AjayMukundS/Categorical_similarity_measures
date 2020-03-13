import pandas as pd
import numpy as np
import category_encoders as ce

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
def Goodall3(data):
 	  data= ordinal_encode(data)
 	  r = data.shape[0]
 	  s = data.shape[1]  
 	  
 	  freq_table= Frequency_table(data)
 	  
 	  freq_rel= freq_table/r
 	  
 	  agreement= []
 	  for i in range(s):
 	      agreement.append(0)
 	  
 	  good3= np.zeros(shape=(r,r))
 	  
 	  for i in range(r-1):
 	      for j in range(1+i, r):
 	          for k in range(s):
 	              c = data.iat[i,k]-1
 	              if (data.iat[i,k] == data.iat[j,k]):
 	                  agreement[k] = 1- freq_rel[c][k]**2
 	              else:
 	                  agreement[k] = 0
 	          if i == j:
 	              good3[i][j]= 0
 	          else:
 	              good3[i][j] = 1-(sum(agreement)/s)
 	              good3[j][i] = good3[i][j]
 	  return(good3)

def ordinal_encode(data):
 	  encoding_data= data.copy()
 	  encoder= ce.OrdinalEncoder(encoding_data)
 	  data_encoded= encoder.fit_transform(encoding_data)
 	  return(data_encoded)
