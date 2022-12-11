import numpy as np
def list2arr(X):
  Xi1=[]
  for xi in X:
    xi1=np.asarray(xi)
    if len(xi1.shape)==3:
      xi1=np.reshape(xi1, [xi1.shape[0], xi1.shape[2]])
    elif len(xi1.shape)==2:
      xi1=np.reshape(xi1, [xi1.shape[0], 1])

    Xi1.append(xi1)
  Xi1=np.asarray(Xi1)
  Xnew=Xi1[0]
  #print("Length Xi1", len(Xi1))
  for j in range(len(Xi1)-1):
    #print(Xi1[j+1].shape)
    Xnew=np.concatenate([Xnew, Xi1[j+1]], axis=0)
  Xnew.shape

  return(Xnew)
