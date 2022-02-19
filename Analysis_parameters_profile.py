# import matplotlib.pyplot as plt
import numpy as np
# from collections import deque
from scipy import stats
import math

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])

afont = {'fontname':'Arial'}

#For Fractal dimensions
end = int(input_list[3])-1
nw_len = int(np.floor(math.log((end-1)/10)/math.log(2)))

nw = np.zeros((nw_len)).astype("int")

for i in range(0,nw_len):
    nw[i] = int(2**(i+1))


#For material ratios
pmp = 0.1
pvv = 0.8

Legends1 = ["Pp","Pv","Pz","P10z","Pa","Pq","Psk","Pku","Fractal dimension"]
Legends2 = ['Pmp','Pvv','Pmc','Pvc']

Prof = np.load('Output_data/profiles_'+str(name)+'.npy')

### FRACTAL DIMENSIONS ###
output=np.zeros((len(Prof[:,0,0]),len(nw)))
for kk in range(0,len(nw)):
    jj=nw[kk]
    wdow = len(Prof[0,:,0])/float(jj)
    sec = 2*jj-1
    rms_part = np.zeros((len(Prof[:,0,0]),sec))
    rms = np.zeros((len(Prof[:,0,0]),1))
    for g in range(0,sec):
    
        low = 0+wdow/float(2)*g
        high = wdow+wdow/float(2)*g
    
        idx = np.arange(int(low),int(high))
    
        Prof_temp = Prof[:,idx,:]
    
        mi = len(Prof_temp[0,:,0])

        for ii in range(0,len(Prof_temp[:,0,0])):
            rj = np.zeros((len(Prof_temp[ii,:,0])))
            fitz = np.polyfit(Prof_temp[ii,:,2], Prof_temp[ii,:,3], 1)
            p1 = np.poly1d(fitz)
            pp = np.zeros((len(Prof_temp[ii,:,3]),2))
            pp[:,0] = Prof_temp[ii,:,2]
            pp[:,1] = p1(Prof_temp[ii,:,2])
            rj[:] = Prof_temp[ii,:,3]-pp[:,1]
            rj_mean = np.mean(rj)
            rms_part[ii,g] = np.sqrt(1/float(mi-2)*np.sum((rj-rj_mean)**2))
    
    rms[:,0] = 1/float(sec)*np.sum(rms_part[:,:],axis=1)
    output[:,kk] = rms[:,0] 
    
    logwindow = np.zeros((nw_len)).astype("float")
    for ij in range(0,nw_len):
        logwindow[ij] = np.log10(len(Prof[0,:,0])/nw[ij])
    D_tot = np.zeros((len(Prof[:,0,0]),1))
for h in range(0, len(D_tot)):
  
    logoutput = np.log10(output[h,:])

    fitz = np.polyfit(logwindow[:], logoutput[:], 1)
    p1 = np.poly1d(fitz)
    pp = np.zeros((len(logwindow[:]),2))
    pp[:,0] = logwindow[:]
    pp[:,1] = p1(logwindow[:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(pp[:,0], pp[:,1])

    D_tot[h,0] = 2-slope
    D_tot_mean = np.mean(D_tot)

### PROFILE PARAMETER CALCULATIONS ###
Prof = np.load('Output_data/profiles_'+str(name)+'.npy')
P_mat = np.zeros((len(Prof[:,:,3]),4))
P_mat_means =np.zeros((4))

intercept1=np.zeros((len(Prof[:,:,3]),1))
slope1=np.zeros((len(Prof[:,:,3]),1))
intercept2=np.zeros((len(Prof[:,:,3]),1))
slope2=np.zeros((len(Prof[:,:,3]),1))
for l in range(0,len(Prof[:,:,3])):
    cfactor = np.arange(len(Prof[l,:,3]))

    fitz = np.polyfit(Prof[l,:,2], Prof[l,:,3], 1)
    p1 = np.poly1d(fitz)
    pp = np.zeros((len(Prof[l,:,3]),2))
    pp[:,0] = Prof[l,:,2]
    pp[:,1] = p1(Prof[l,:,2])
    slope1[l,0], intercept1[l,0], r_value, p_value, std_err = stats.linregress(pp[:,0], pp[:,1])
    Prof[l,:,3] = Prof[l,:,3]-cfactor*slope1[l,0]

    fitz = np.polyfit(Prof[l,:,2], Prof[l,:,3], 1)
    p1 = np.poly1d(fitz)
    pp = np.zeros((len(Prof[l,:,3]),2))
    pp[:,0] = Prof[l,:,2]
    pp[:,1] = p1(Prof[l,:,2])
    slope2[l,0], intercept2[l,0], r_value, p_value, std_err = stats.linregress(pp[:,0], pp[:,1])
    Prof[l,:,3] = Prof[l,:,3]-intercept2[l,0]

Prof[:,:,3] = Prof[:,:,3]*ps

Pp_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pp_angle[i,0] = np.max(Prof[i,:,3])
    
Pv_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pv_angle[i,0] = np.min(Prof[i,:,3])
    
Pz_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pz_angle[i,0] = (abs(np.min(Prof[i,:,3]))+abs(np.max(Prof[i,:,3])))
    
    P10z_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    findx1 = np.sort(Prof[i,:,3])[-10:]
    findx2 = np.sort(Prof[i,:,3])[0:10]
    P10z_angle[i,0] = (abs(np.sum(findx1-findx2)))/10
    
Pa_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pa_angle[i,0] = np.mean(abs(Prof[i,:,3]))

Pq_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pq_angle[i,0] = np.sqrt(np.mean(Prof[i,:,3]**2))
    
Psk_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Psk_angle[i,0] =(np.mean((Prof[i,:,3])**3))/float(Pq_angle[i]**3)
    
Pku_angle = np.zeros((len(Prof[:,0,0]),1))
for i in range(0,len(Prof[:,0,0])):
    Pku_angle[i,0] =(np.mean((Prof[i,:,3])**4))/float(Pq_angle[i]**4)

Pp_mean = np.mean(Pp_angle)
Pv_mean = np.mean(Pv_angle)
Pz_mean = np.mean(Pz_angle)
P10z_mean = np.mean(P10z_angle)
Pa_mean = np.mean(Pa_angle)
Pq_mean = np.mean(Pq_angle)
Psk_mean = np.mean(Psk_angle)
Pku_mean = np.mean(Pku_angle)

zerl = len(Prof[:,:,3])
P_angle = [Pp_angle[:,0],Pv_angle[:,0],Pz_angle[:,0],P10z_angle[:,0],Pa_angle[:,0],Pq_angle[:,0],Psk_angle[:,0],Pku_angle[:,0],D_tot[:,0]]
np.save('Output_data/P_angle_'+str(name)+'.npy', P_angle)
P_angle_means=[Pp_mean,Pv_mean,Pz_mean,P10z_mean,Pa_mean,Pq_mean,Psk_mean,Pku_mean,D_tot_mean]
np.save('Output_data/P_angle_means_'+str(name)+'.npy', P_angle_means)

##Material ratio stuff

Material=Prof[:,:,3]
M_sort=-np.sort(-Material, axis=1)
length=len(M_sort[0,:])

for i in range(0,len(Prof[:,0,0])):
    
    maxx=np.max(M_sort[i,:])
    minn=np.min(M_sort[i,:])
    
    cutmp=np.int(np.round(length*pmp))
    cutvv=np.int(np.round(length*pvv))
    
    temp = M_sort[i,:]-minn
    
    P_mat[i,0]=np.sum(temp[0:cutmp])-temp[cutmp]*cutmp
    
    P_mat[i,1]=temp[cutvv]*(length-cutvv)-np.sum(temp[cutvv:-1])
    
    P_mat[i,2]=(temp[cutmp]-temp[cutvv])*cutmp+np.sum(temp[cutmp:cutvv])-(temp[cutvv]-temp[-1])*(cutvv-cutmp)
    
    P_mat[i,3]=(temp[cutmp]-temp[cutvv])*length-P_mat[i,2]
    
    O_areal= (temp[cutvv]-temp[-1])*length-P_mat[i,1]
    
    Tot_test = np.sum(temp)
    Tot_areal = P_mat[i,0]+P_mat[i,2]+O_areal
    
P_mat_means[0]=np.mean(P_mat[:,0])
P_mat_means[1]=np.mean(P_mat[:,1])
P_mat_means[2]=np.mean(P_mat[:,2])
P_mat_means[3]=np.mean(P_mat[:,3])

np.save('Output_data/P_mat_'+str(name)+'.npy', P_mat)
np.save('Output_data/P_mat_means_'+str(name)+'.npy', P_mat_means)