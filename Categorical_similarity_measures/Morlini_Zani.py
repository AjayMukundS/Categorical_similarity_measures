import pandas as pd
import numpy as np
import math

def Morlini(data):
 	  r = data.shape[0]
 	  s = data.shape[1]  
 	  
 	  num_cat= []
 	  
 	  for col_num in range(len(data.columns)):
 	      col_name= data.columns[col_num]
 	      categories = list(data[col_name].unique())
 	      num_cat.append(len(categories))
 	  
 	  data_dummy= pd.get_dummies(data, columns= data.columns)
 	  
 	  n= data_dummy.shape[0]
 	  hs= data_dummy.shape[1]  
 	  
 	  nsv= []
 	  fsv2= []
 	  for i in range(hs):
 	      nsv.append(data_dummy[data_dummy.columns[i]].sum())
 	      fsv2.append(np.log(1/(nsv[i]/n)**2))
 	  
 	  
 	  agreement= []
 	  for i in range(hs):
 	      agreement.append(0)
 	  
 	  E= np.zeros(shape=(n,n))
 	  
 	  for i in range(n-1):
 	      for j in range(1+i, n):
 	          for k in range(hs):
 	              if (data_dummy.iat[i,k] == 1 and data_dummy.iat[j,k]== 1):
 	                  agreement[k] = 1
 	              else:
 	                  agreement[k] = 0
 	          E[i][j]= np.matmul(fsv2, agreement)
 	          E[j][i] = E[i][j]

 	  cum= []
 	  for i in range(len(num_cat)):
 	      if(i== 0):
 	          cum.append(num_cat[i])
 	      else:
 	          cum.append(cum[i-1]+ num_cat[i])
 	          
 	  F= np.zeros(shape=(n,n))
 	  
 	  for i in range(n-1):
 	      for j in range(1+i, n):
 	          v= 0
 	          agreement= []
 	          for value in range(hs):
 	              agreement.append(0)
 	          for k in range(s):
 	              for t in range((v),cum[k]):
 	                  if (data_dummy.iat[i,t] == 0 and data_dummy.iat[j,t]== 1):
 	                      for val in range((v), cum[k]):
 	                          agreement[val]= 1
 	              v= cum[k]
 	          F[i][j]= np.matmul(agreement, fsv2)
 	          F[j][i]= F[i][j]
 	          
 	  morlini= 1- E/ (E+F)
 	  
 	  morlini= np.nan_to_num(morlini)
 	  
 	  return(morlini)
