import pandas as pd
import numpy as np
import category_encoders as ce

def ordinal_encode(data):
 	  encoding_data= data.copy()
 	  encoder= ce.OrdinalEncoder(encoding_data)
 	  data_encoded= encoder.fit_transform(encoding_data)
 	  return(data_encoded)
def Overlap(data):
 	  data= ordinal_encode(data)
 	  r = data.shape[0]
 	  s = data.shape[1]
 	      
 	  agreement= []
 	  for i in range(s):
 	      agreement.append(0)
 	  
 	  overlap= np.zeros(shape=(r,r))
 	  
 	  for i in range(r-1):
 	      for j in range(1+i, r):
 	          for k in range(s):
 	              if data.iat[i, k] == data.iat[j, k]:
 	                  agreement[k] = 1
 	              else:
 	                  agreement[k] = 0
 	          if sum(agreement)!= 0 :
 	              overlap[i][j] = 1-1/s*(sum(agreement))
 	              overlap[j][i] = overlap[i][j]
 	          else:
 	              overlap[i][j]= 1
 	              overlap[j][i] = overlap[i][j]
 	  return(overlap)
