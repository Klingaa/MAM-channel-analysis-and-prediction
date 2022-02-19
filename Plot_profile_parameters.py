import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque
from scipy import stats
from mpl_toolkits import mplot3d

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])

afont = {'fontname':'Arial'}

#For material ratios
pmp = 0.1
pvv = 0.8

Legends1 = ["Pp","Pv","Pz","P10z","Pa","Pq","Psk","Pku","Fractal dimension"]
Legends2 = ['Pmp','Pvv','Pmc','Pvc']


##### COMBINING AND PLOTTING PROFILE PARAMETERS #####

P_1t = np.load('Output_data/P_angle_'+str(name)+'.npy')
P_1_means=np.load('Output_data/P_angle_means_'+str(name)+'.npy')

t_length = len(P_1t[0])

P_tot=P_1t.T

P_tot_means=P_1_means

####Rotation####
lok_pa2=np.where(P_tot_means[4] > P_tot[:,4])
rotation_ini = int(t_length-np.min(lok_pa2))
for n in range(0,9):
    tester = deque(P_tot[:,n])
    tester.rotate(rotation_ini)
    P_tot[:,n]=np.asarray(tester)
 
lok_pa=np.where(P_tot_means[4] < P_tot[:,4])
ang_middle = int(round((np.max(lok_pa)-np.min(lok_pa))/2+np.min(lok_pa)))
#np.save('ang_middle_'+str(name)+'.npy', ang_middle)
rotation= int(t_length/2-ang_middle)
for n in range(0,9):
    tester = deque(P_tot[:,n])
    tester.rotate(rotation)
    P_tot[:,n]=np.asarray(tester)

np.save('Output_data/P_tot_'+str(name)+'.npy', P_tot)
np.save('Output_data/P_tot_means_'+str(name)+'.npy', P_tot_means)

for m in range(0,8):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for j in range(0,t_length):
        sc = ax.scatter(j/(float(t_length)-1)*360, P_tot[j,m], c = 'k', s = 5)
    line=plt.plot([0,360],[P_tot_means[m],P_tot_means[m]],c='r',linestyle="--",linewidth=2)
    ax.grid(True)
    ax.set_xlabel('$\\beta$ [$\degree$]', labelpad=14, fontsize=20)
    ax.set_ylabel(str(Legends1[m]) +' [$\mu m$]', labelpad=14, fontsize=20)
    ax.set_xlim(0, 360)
#       ax.set_ylim(0, 1.05*max(P_angle[m]))
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    plt.legend([line[0], sc], ['Average '+str(Legends1[m]), str(Legends1[m])+'-values'],fontsize = 20,loc='upper right')
    plt.xticks(np.arange(0, 360+1, 45))
    plt.show()
    plt.savefig('Output_plots/'+str(name)+'_'+str(Legends1[m])+'_'+str(np.round(P_tot_means[m],3))+'.tiff')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    for k in range(0,t_length):
        sc1 = ax.scatter(k/(float(t_length)-1)*np.pi*2, P_tot[k,m], c = 'k', s=10)
    r = np.full(10000, P_tot_means[m])
    theta = np.arange(0,2*np.pi,2*np.pi/10000)
    ax.plot(theta, r,c='r',linestyle="--",linewidth=2)
#    ax.set_rmax(250)
    ax.grid(True)
    ax.tick_params(labelsize=25)
    ax.set_theta_zero_location("S")
    plt.show()
    plt.savefig('Output_plots/'+str(name)+'_'+str(Legends1[m])+'_'+str(np.round(P_tot_means[m],3))+'_polar.tiff')

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
for j in range(0,t_length):
    sc = ax.scatter(j/(float(t_length)-1)*360, P_tot[j,8], c = 'k', s = 5)
line=plt.plot([0,360],[P_tot_means[8],P_tot_means[8]],c='r',linestyle="--",linewidth=2)
ax.grid(True)
ax.set_xlabel('$\\beta$ [$\degree$]', labelpad=14, fontsize=20)
ax.set_ylabel('Fractal dimension [-]', labelpad=14, fontsize=20)
ax.set_xlim(0, 360)
#ax.set_ylim(0, 1.05*max(P_angle[m]))
ax.tick_params(labelsize=20)
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
plt.legend([line[0], sc], ['Average fractal dimension', 'Fractal dimension values'],fontsize = 20,loc='upper right')
plt.xticks(np.arange(0, 360+1, 45))
plt.show()
plt.savefig('Output_plots/'+str(name)+'_FractalDimension_'+str(np.round(P_tot_means[8],3))+'.tiff')
    
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')
for k in range(0,t_length):
    sc1 = ax.scatter(k/(float(t_length)-1)*np.pi*2, P_tot[k,8], c = 'k', s=10)
r = np.full(10000, P_tot_means[8])
theta = np.arange(0,2*np.pi,2*np.pi/10000)
ax.plot(theta, r,c='r',linestyle="--",linewidth=2)
ax.set_ylim(1,2)
ax.set_yticks(np.arange(1,2,0.2))
ax.grid(True)
ax.tick_params(labelsize=25)
ax.set_theta_zero_location("S")
plt.show()
plt.savefig('Output_plots/'+str(name)+'_FractalDimension_'+str(np.round(P_tot_means[8],3))+'_polar.tiff')

##### COMBINING AND PLOTTING MATERIAL RATIO PARAMTERS #####

P_1mat = np.transpose(np.load('Output_data/P_mat_'+str(name)+'.npy'))

P_1mat_means=np.load('Output_data/P_mat_means_'+str(name)+'.npy')

zerl_1mat = len(P_1mat[0])

mat_length=zerl_1mat

P_tot_mat=P_1mat.T

P_tot_meansmat=P_1mat_means


for n in range(0,4):
    tester = deque(P_tot_mat[:,n])
    tester.rotate(rotation)
    P_tot_mat[:,n]=np.asarray(tester)

for m in range(0,4):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for j in range(0,mat_length):
        sc = ax.scatter(j/(float(mat_length)-1)*360, P_tot_mat[j,m]/(1000), c = 'k', s = 5)
    line=plt.plot([0,360],[P_tot_meansmat[m]/(1000),P_tot_meansmat[m]/(1000)],c='r',linestyle="--",linewidth=2)
    ax.grid(True)
    ax.set_xlabel('$\\beta$ [$\degree$]', labelpad=14, fontsize=20)
    ax.set_ylabel(str(Legends2[m]) +' [$mm$]', labelpad=14, fontsize=20)
    ax.set_xlim(0, 360)
#       ax.set_ylim(0, 1.05*max(P_angle[m]))
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    plt.legend([line[0], sc], ['Average '+str(Legends2[m]), str(Legends2[m])+'-values'],fontsize = 20,loc='upper right')
    plt.xticks(np.arange(0, 360+1, 45))
    plt.show()
    plt.savefig('Output_plots/'+str(name)+'_'+str(Legends2[m])+'_'+str(np.round(P_tot_meansmat[m],3))+'.tiff')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    for k in range(0,t_length):
        sc1 = ax.scatter(k/(float(t_length)-1)*np.pi*2, P_tot_mat[k,m]/(1000), c = 'k', s=10)
    r = np.full(10000, P_tot_meansmat[m]/(1000))
    theta = np.arange(0,2*np.pi,2*np.pi/10000)
    ax.plot(theta, r,c='r',linestyle="--",linewidth=2)
#    ax.set_rmax(250)
    ax.grid(True)
    ax.tick_params(labelsize=25)
    ax.set_theta_zero_location("S")
    plt.show()
    plt.savefig('Output_plots/'+str(name)+'_'+str(Legends2[m])+'_'+str(np.round(P_tot_meansmat[m],3))+'_polar.tiff')

#Save all computed results in txt file
file = open('Output_txt/Profile_parameter_analysis'+str(name)+'_'+str(ps)+'.txt',"w+")

file.write("_____Profile Parameters_____\n\n")
file.write(str(Legends1[0])+" avg is "+str(P_tot_means[0])+" mu\n")
file.write(str(Legends1[1])+" avg is "+str(P_tot_means[1])+" mu\n")
file.write(str(Legends1[2])+" avg is "+str(P_tot_means[2])+" mu\n")
file.write(str(Legends1[3])+" avg is "+str(P_tot_means[3])+" mu\n")
file.write(str(Legends1[4])+" avg is "+str(P_tot_means[4])+" mu\n")
file.write(str(Legends1[5])+" avg is "+str(P_tot_means[5])+" mu\n")
file.write(str(Legends1[6])+" avg is "+str(P_tot_means[6])+"\n")
file.write(str(Legends1[7])+" avg is "+str(P_tot_means[7])+"\n")
file.write(str(Legends1[8])+" avg is "+str(P_tot_means[8])+"\n")
file.write("\n")
file.write("_____Material Volume Parameters mr1 = "+str(pmp)+" mr2 = "+str(pvv)+"_____\n\n")
file.write(str(Legends2[0])+" avg is "+str(P_tot_meansmat[0])+" mu\n")
file.write(str(Legends2[1])+" avg is "+str(P_tot_meansmat[1])+" mu\n")
file.write(str(Legends2[2])+" avg is "+str(P_tot_meansmat[2])+" mu\n")
file.write(str(Legends2[3])+" avg is "+str(P_tot_meansmat[3])+" mu\n")

file.close() 

#save profile data array
profiles = np.asarray(P_1t).T
profile_name=str("Pp Pv pz P10z Pa Pq Psk Pku FratalDimension")
np.savetxt('Output_txt/Profile_data_'+str(name)+'_'+str(ps)+'.txt', profiles, header=profile_name, comments='')