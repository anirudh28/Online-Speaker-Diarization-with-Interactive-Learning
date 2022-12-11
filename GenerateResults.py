import numpy as np
import matplotlib.pyplot as plt
import pathlib
### RESULTS


path=pathlib.Path(__file__).parent.absolute()

################### Epireward 0.1
print('\n 5 Users EpiReward 0.1    \n')
data_LinUCB = np.loadtxt(str(path)+"/User5epi0.1/data_LinUCB_5_0.1_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User5epi0.1/data_BerlinUCB_5_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User5epi0.1/data_BerlinUCB_KNN_5_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User5epi0.1/data_BerlinUCB_KMeans_5_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User5epi0.1/data_BerlinUCB_GMM_5_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 5, epiReward = 0.1')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


###################################
print('\n 10 Users EpiReward 0.1    \n')

data_LinUCB = np.loadtxt(str(path)+"/User10epi0.1/data_LinUCB_10_0.1_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User10epi0.1/data_BerlinUCB_10_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User10epi0.1/data_BerlinUCB_KNN_10_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User10epi0.1/data_BerlinUCB_KMeans_10_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User10epi0.1/data_BerlinUCB_GMM_10_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 10, epiReward = 0.1')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


############################################################

print('\n 15 Users EpiReward 0.1    \n')

data_LinUCB = np.loadtxt(str(path)+"/User15epi0.1/data_LinUCB_15_0.1_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User15epi0.1/data_BerlinUCB_15_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User15epi0.1/data_BerlinUCB_KNN_15_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User15epi0.1/data_BerlinUCB_KMeans_15_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User15epi0.1/data_BerlinUCB_GMM_15_0.1_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 15, epiReward = 0.1')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)

#######################################################################

################### Epireward 0.5
print('\n 5 Users EpiReward 0.5   \n')
data_LinUCB = np.loadtxt(str(path)+"/User5epi0.5/data_LinUCB_5_0.5_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User5epi0.5/data_BerlinUCB_5_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User5epi0.5/data_BerlinUCB_KNN_5_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User5epi0.5/data_BerlinUCB_KMeans_5_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User5epi0.5/data_BerlinUCB_GMM_5_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 5, epiReward = 0.5')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


###################################
print('\n 10 Users EpiReward 0.5   \n')

data_LinUCB = np.loadtxt(str(path)+"/User10epi0.5/data_LinUCB_10_0.5_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User10epi0.5/data_BerlinUCB_10_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User10epi0.5/data_BerlinUCB_KNN_10_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User10epi0.5/data_BerlinUCB_KMeans_10_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User10epi0.5/data_BerlinUCB_GMM_10_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 10, epiReward = 0.5')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


############################################################

print('\n 15 Users EpiReward 0.5    \n')

data_LinUCB = np.loadtxt(str(path)+"/User15epi0.5/data_LinUCB_15_0.5_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User15epi0.5/data_BerlinUCB_15_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User15epi0.5/data_BerlinUCB_KNN_15_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User15epi0.5/data_BerlinUCB_KMeans_15_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User15epi0.5/data_BerlinUCB_GMM_15_0.5_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 15, epiReward = 0.5')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


###########################################################

################### Epireward 0.01
print('\n 5 Users EpiReward 0.01   \n')
data_LinUCB = np.loadtxt(str(path)+"/User5epi0.01/data_LinUCB_5_0.01_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User5epi0.01/data_BerlinUCB_5_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User5epi0.01/data_BerlinUCB_KNN_5_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User5epi0.01/data_BerlinUCB_KMeans_5_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User5epi0.01/data_BerlinUCB_GMM_5_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 5, epiReward = 0.01')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


###################################
print('\n 10 Users EpiReward 0.01   \n')

data_LinUCB = np.loadtxt(str(path)+"/User10epi0.01/data_LinUCB_10_0.01_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User10epi0.01/data_BerlinUCB_10_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User10epi0.01/data_BerlinUCB_KNN_10_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User10epi0.01/data_BerlinUCB_KMeans_10_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User10epi0.01/data_BerlinUCB_GMM_10_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 10, epiReward = 0.01')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


############################################################

print('\n 15 Users EpiReward 0.01    \n')

data_LinUCB = np.loadtxt(str(path)+"/User15epi0.01/data_LinUCB_15_0.01_10_0.csv", delimiter=",")
reward_cumulative_LinUCB = data_LinUCB[0]
accuracy_cumulative_LinUCB = data_LinUCB[1]
final_acc_LinUCB = accuracy_cumulative_LinUCB[-1]
t_array = data_LinUCB[2]
# t = len(t_array) # total timesteps

data_BerlinUCB = np.loadtxt(str(path)+"/User15epi0.01/data_BerlinUCB_15_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB = data_BerlinUCB[0]
accuracy_cumulative_BerlinUCB = data_BerlinUCB[1]
final_acc_BerlinUCB = accuracy_cumulative_BerlinUCB[-1]
t_array = data_BerlinUCB[2]


data_BerlinUCB_KNN = np.loadtxt(str(path)+"/User15epi0.01/data_BerlinUCB_KNN_15_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[0]
accuracy_cumulative_BerlinUCB_KNN = data_BerlinUCB_KNN[1]
final_acc_BerlinUCB_KNN = accuracy_cumulative_BerlinUCB_KNN[-1]
t_array = data_BerlinUCB_KNN[2]

data_BerlinUCB_KMeans = np.loadtxt(str(path)+"/User15epi0.01/data_BerlinUCB_KMeans_15_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[0]
accuracy_cumulative_BerlinUCB_KMeans = data_BerlinUCB_KMeans[1]
final_acc_BerlinUCB_KMeans = accuracy_cumulative_BerlinUCB_KMeans[-1]
t_array = data_BerlinUCB_KMeans[2]

data_BerlinUCB_GMM = np.loadtxt(str(path)+"/User15epi0.01/data_BerlinUCB_GMM_15_0.01_10_0.csv", delimiter=",")
reward_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[0]
accuracy_cumulative_BerlinUCB_GMM = data_BerlinUCB_GMM[1]
final_acc_BerlinUCB_GMM = accuracy_cumulative_BerlinUCB_GMM[-1] 
t_array = data_BerlinUCB_GMM[2]

plt.figure()
plt.plot(t_array, reward_cumulative_LinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB) 
plt.plot(t_array, reward_cumulative_BerlinUCB_GMM) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KNN) 
plt.plot(t_array, reward_cumulative_BerlinUCB_KMeans) 
plt.legend(["LinUCB","BerlinUCB","BerlinUCB (GMM)","BerlinUCB (KNN)","BerlinUCB (K Means)"])
plt.title('reward vs timestep \n for speakers = 15, epiReward = 0.01')
plt.xlabel('timestep')
plt.ylabel('reward')
plt.show()

print('\nvalues for final accuracy = (total reward)/(total timestep) \n')
print('LinUCB :              ', final_acc_LinUCB)
print('BerlinUCB :           ', final_acc_BerlinUCB)
print('BerlinUCB (GMM) :     ', final_acc_BerlinUCB_GMM)
print('BerlinUCB (KNN) :     ', final_acc_BerlinUCB_KNN)
print('BerlinUCB (K Means) : ', final_acc_BerlinUCB_KMeans)


