# Online-Speaker-Diarization-with-Interactive-Learning




Team TeNET
Team Members:
1) Anirudh Garg, Department of Electrical Engineering
2) Anubhav Satpathy, Department of Electrical Engineering
3) Nitish Vikas Deshpande, Department of Electrical Engineering


Project Topic: Online Speaker Diarization with Interactive Learning

We implement 5 methods for Online Speaker Diarization.

a) Linear Upper Confidence Bound Algorithm (LinUCB) ---> Linucb.py

b)  Background Episodically Rewarded Linear Upper Confidence Bound Algorithm (BerlinUCB) ---> BerlinUCB.py

c) BerlinUCB with self supervision using K Nearest Neighbours (KNN) ---> BerlinUCB_KNN.py

d) BerlinUCB with self supervision using Gaussian Mixture Models (GMM)  ---> BerlinUCB_GMM.py

e) BerlinUCB with self supervision using K Means clustering   --->  BerlinUCB_Kmeans.py


The data files are included in the MAT_FILES folder.  It consists of 9 .mat files. Each file corresponds to no of speakers from the set {5, 10, 15} and epiReward i.e, the revealing probability from the set {0.01, 0.1, 0.5}. 

Instructions to use this repository.

1) In main.py file, we provide an example of using the data corresponding to 15 speakers and epiReward=0.1  (MFCC_15_0.1_10_0.mat).  We apply the BerlinUCB Model on this data. The results corresponding to each file and each model can be obtained by changing the file name and the model name in main.py code file. All results are then saved in csv format.

2) The results are categorized in 9 folders (User5epi0.01, User5epi0.1, User5epi0.5, User10epi0.01, User10epi0.1, User10epi0.5, User15epi0.01, User15epi0.1, User15epi0.5 ).  Each folder has 5 files corresponding to the 5 models.  


3) The GenerateResults.py file is used to generate the plots of Reward vs Timestep for each of the 9 .mat files. These plots are provided in .eps format in the Plots folder.
