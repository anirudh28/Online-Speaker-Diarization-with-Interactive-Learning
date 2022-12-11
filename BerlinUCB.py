import numpy as np
from check_existing import check_existing
def BerlinUCB(mat):

  rec=mat['data'][0,0]['rec']
  dim=int(mat['data'][0,0]['dim'])
  t=int(mat['data'][0,0]['t'])
  full_y=mat['data'][0,0]['full_y']
  y=mat['data'][0,0]['y']
  useNN=mat['data'][0,0]['useNN']
  nOptions=mat['data'][0,0]['nOptions']
  epiReward=mat['data'][0,0]['epiReward']
  oracle=mat['data'][0,0]['oracle']
  nPiece=mat['data'][0,0]['nPiece']


  ucb_alpha = 0.1;
  A = []
  b = []

  
  nArms = 1
  A.append(np.eye(dim))
  b.append(np.zeros([dim,1 ]))
  labls = ['new']
  labln = [0]

  y_true = full_y
  x = rec

  reward = 0;
  reward_cumulative = [];
  accuracy_cumulative = [];

  for ts in range(t):
      print("Time step is ", ts)

      labl = y[ts]
      labl_true = y_true[ts]
      feat_vec= np.expand_dims(x[:, ts] , axis=1)    ## Make it a column vector

      stillWrong = 0
      stillCorrect = 0

      p_ts=[]

      for i in range(nArms):
        print(i)
        theta=np.dot(np.linalg.inv(A[i]),b[i])
        p_ts_arm=  np.dot(theta.T,feat_vec)+ucb_alpha*((np.dot(feat_vec.T, np.dot(np.linalg.inv(A[i]),feat_vec)))**0.5)
        print(p_ts_arm)
        p_ts.append(p_ts_arm)
      

      pred=np.argmax(p_ts)
      print("Pred is",pred)
      
      
      if not check_existing(labls, labl ) and (labl != '-1'):
        nArms=nArms+1
        labls.append(labl)
        labln.append(nArms-1)
        A.append(np.eye(dim))
        b.append(np.zeros([dim,1 ]))
        if pred==1:
          stillCorrect=1

      if not stillWrong and (stillCorrect or labls[labln.index(pred)]== labl_true ):
        reward=reward+1
      
      reward_cumulative.append(reward)
      accuracy=reward/(ts+1)
      print("Accuracy is", accuracy, "\n")
      accuracy_cumulative.append(accuracy)
      
      if (labl != '-1') and (labls[labln.index(pred)]== labl):                                      ## For kmeans or knn or gaussian mixture, add the case of labl == -1
          feedback_t=1
      else:
          feedback_t=0
        

      A[pred]=A[pred]+ np.dot(feat_vec, feat_vec.T)              ## Outside the if condition, different from the linUCB case
      b[pred]=b[pred]+feedback_t*feat_vec
      
      
  return reward_cumulative, accuracy_cumulative, t

