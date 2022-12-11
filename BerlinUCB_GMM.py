import numpy as np
from check_existing import check_existing
def BerlinUCB_GMM(mat):

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

  nArms = 1
  A.append(np.eye(dim))
  b.append(np.zeros([dim,1 ]))
  labls = ['new']
  labln = [0]      

  y_true = full_y
  x = rec ## x = (dim,t) = MFCC matrix

  kms=x[:,0:nArms].T
  kmv=np.ones([nArms, dim]) 
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
        kmv=np.concatenate((kmv, np.ones([1,dim])), axis=0)
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
        kmv[assignment, :]= np.std(hbk[assignment], axis=0) 
        
        if labls[labln.index(pred)]== labl:
          feedback_t=1
        else:
          feedback_t=0
        A[pred]=A[pred]+ np.dot(feat_vec, feat_vec.T)             
        b[pred]=b[pred]+feedback_t*feat_vec

      elif (labl=='-1'):   ## decide feedback, update b, keep A unchanged
        nlls = []
        for i in range(nArms):
          nll = 0  ## negative log likelihood
          for j in range(dim):
            nll = nll + 0.5*(((feat_vec[j,0]-kms[i,j])/kmv[i,j])**2) + np.log(kmv[i,j])  ## no need to add const term = log(sqrt(2*pi))
          nlls.append(nll) 

        pred_self = np.argmin(nlls)  ## whichever gives lowest nll (i.e. maximum likelihood), choose that as pred_self 
        if pred_self== pred:
          feedback_t=1
        else:
          feedback_t=0
        b[pred]=b[pred]+feedback_t*feat_vec

      
  return reward_cumulative, accuracy_cumulative, t

