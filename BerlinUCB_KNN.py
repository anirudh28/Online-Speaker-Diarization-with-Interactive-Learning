import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from check_existing import check_existing
from list2arr import list2arr

def BerlinUCB_KNN(mat):

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


  ucb_alpha = 0.1
  A = []
  b = []
  k=5   # k in KNN algorithm

  
  nArms = 1
  A.append(np.eye(dim))
  b.append(np.zeros([dim,1 ]))
  labls = ['new']
  labln = [0]

  y_true = full_y
  x = rec

  kms=x[:,0:nArms].T
  kmn=np.zeros([nArms, 1])
  idx=np.arange(nArms)

  hbk=[]

  for i in range(nArms):
    hbk.append([])



  reward = 0;
  reward_cumulative = [];
  accuracy_cumulative = [];

  for ts in range(t):
      print("Time step is ", ts)

      labl = y[ts]
      #print(labl)
      labl_true = y_true[ts]
      feat_vec= np.expand_dims(x[:, ts] , axis=1)    ## Make it a column vector

      stillWrong = 0
      stillCorrect = 0

      p_ts=[]

      for i in range(nArms):
        #print(i)
        theta=np.dot(np.linalg.inv(A[i]),b[i])
        p_ts_arm=  np.dot(theta.T,feat_vec)+ucb_alpha*(np.dot(feat_vec.T, np.dot(np.linalg.inv(A[i]),feat_vec)))**0.5
        #print(p_ts_arm)
        p_ts.append(p_ts_arm)
      

      pred=np.argmax(p_ts)
      #print("Pred is",pred)
      
      
      if not check_existing(labls, labl ) and (labl != '-1'):
        nArms=nArms+1
        labls.append(labl)
        labln.append(nArms-1)
        kms=np.concatenate((kms,feat_vec.T), axis=0)
        kmn=np.concatenate((kmn, np.zeros([1,1])), axis=0)
        idx=np.arange(nArms)
        hbk.append([])
        A.append(np.eye(dim))
        b.append(np.zeros([dim,1 ]))
        if pred==1:
          stillCorrect=1
      #print(nArms)
      if not stillWrong and (stillCorrect or labls[labln.index(pred)]== labl_true ):
        reward=reward+1
      
      reward_cumulative.append(reward)
      accuracy=reward/(ts+1)
      print("Accuracy is", accuracy, "\n")
      accuracy_cumulative.append(accuracy)
      
      if (labl != '-1'):                                      ## For kmeans or knn or gaussian mixture, add the case of labl == -1
        assignment= labln[labls.index(labl)]
        #print("Assigning ", assignment)
        kmn[assignment]=kmn[assignment]+1
        hbk[assignment].append(feat_vec.T)
        kms[assignment, :]= np.mean(hbk[assignment], axis=0)
        
        if labls[labln.index(pred)]== labl:
          feedback_t=1
        else:
          feedback_t=0
        A[pred]=A[pred]+ np.dot(feat_vec, feat_vec.T)             
        b[pred]=b[pred]+feedback_t*feat_vec
      elif (labl=='-1'):
        knns=[]
        knny=[]
        for i in range(nArms-1):
          knns.append(hbk[(i+1)])
          #print(kmn[(i+1)])
          knny.append((i+1)*np.ones([int(kmn[(i+1)]),1]))
        #print(knns)
        if len(knns)>0:
          knns=list2arr(knns)
          flag_length_zero=0
        else:
          flag_length_zero=1
        if len(knny) >0:
          knny=list2arr(knny)
        if flag_length_zero==0:
          if  knns.shape[0] < k+1:
            model1=KNeighborsClassifier(n_neighbors=1)
            #print("KMS shape", kms[1:, :].shape)
            model1.fit(kms[1:, :], np.arange(1, kms.shape[0]))  
            pred_self=  model1.predict(feat_vec.T)
          else:
            model2=KNeighborsClassifier(n_neighbors=k)
            model2.fit(knns,knny)  
            pred_self=model2.predict(feat_vec.T)
          if pred_self== pred:
            feedback_t=1
          else:
            feedback_t=0
          b[pred]=b[pred]+feedback_t*feat_vec



        

      
      
      
  return reward_cumulative, accuracy_cumulative, t

