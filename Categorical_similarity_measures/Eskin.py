import pandas as pd
import numpy as np
import category_encoders as ce


def ordinal_encode(data):
 	  encoding_data= data.copy()
 	  encoder= ce.OrdinalEncoder(encoding_data)
 	  data_encoded= encoder.fit_transform(encoding_data)
 	  return(data_encoded)
def Eskin(data):
 	  data= ordinal_encode(data)
 	  r = data.shape[0]
 	  s = data.shape[1]
 	  
 	  num_cat= []
 	  
 	  for col_num in range(len(data.columns)):
 	      col_name= data.columns[col_num]
 	      categories = list(data[col_name].unique())
 	      num_cat.append(len(categories))
 	      
 	  agreement= []
 	  for i in range(s):
 	      agreement.append(0)
 	  
 	  eskin= np.zeros(shape=(r,r))
 	  
 	  for i in range(r-1):
 	      for j in range(1+i, r):
 	          for k in range(s):
 	              if data.iat[i, k] == data.iat[j, k]:
 	                  agreement[k] = 1
 	              else:
 	                  agreement[k] = num_cat[k]**2/(num_cat[k]**2 + 2)
 	          eskin[i][j] = (s/sum(agreement))-1
 	          eskin[j][i] = eskin[i][j]
 	  return(eskin)
