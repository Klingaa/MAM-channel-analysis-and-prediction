# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:24:41 2019

@author: cgkli
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.pyplot import gca

ps = 15.8304

afont = {'fontname':'Arial'}

names = ["_0_","_15_","_30_","_45_","_60_","_75_","_90_"]

nlegends = ["$\\alpha=0\degree$","$\\alpha=15\degree$","$\\alpha=30\degree$","$\\alpha=45\degree$","$\\alpha=60\degree$","$\\alpha=75\degree$","$\\alpha=90\degree$"]

path = ""

location = [str(path)+"0",str(path)+"15",str(path)+"30",str(path)+"45",str(path)+"60",str(path)+"75",str(path)+"90"]

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111)
plt.xticks(np.arange(0, 360+1, 45))
plt.show()

symbols = ['o','^','s','x','D','+','v']
colors = ['k','r','g','b','y','c','m']

vinkel = [0,15,30,45,60,75,90]

Pa_angle_x = np.column_stack((0,0,0))

Pa_list = []

def mapr(r):
        return 200 - r
zerlcount = np.zeros((7))

for i in range(0,7):

    name = i
    komp = np.load(str(location[i])+'/P_tot_s01_image_slices.npy')
    # komp_mean = np.load(str(location[i])+'/P_tot_means_s01_image_slices.npy')
    zerlcount[i] = len(komp[:,4])
    zerl=int(zerlcount[i])
    Pa_angle = komp[:,4]
    # Pa_mean = komp_mean[4]
    
    ang1 = 0
    ang2 = 360
    
    xcor = np.zeros((zerl))
    ycor = np.zeros((zerl))
    
    for k in range(0,zerl):
        xcor[k] = k/(float(zerl)-1)*360
        ycor[k] = Pa_angle[k]
        sc1=ax1.scatter(xcor[k], ycor[k], c = colors[i], marker=symbols[i], s = 30)
    sc111=ax1.scatter(xcor[0], ycor[0], c = colors[i], marker=symbols[i], s=30, label =str(nlegends[i]))
    ax1.grid(True)
    ax1.set_xlabel('$\\beta$ [$\degree$]', labelpad=14, fontsize=20, **afont)
    ax1.set_ylabel('Pa [$\mu m$]', labelpad=14, fontsize=20, **afont)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0,60)
    ax1.tick_params(labelsize=20)
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    a = gca()
    a.set_xticklabels(a.get_xticks(), **afont)
    a.set_yticklabels(a.get_yticks(), **afont)
    ax1.legend(fontsize = 15,loc='upper right')
    
    
    xcor = np.zeros((zerl))
    ycor = np.zeros((zerl))
    alph = np.zeros((zerl))
    
    for k in range(0,zerl):
        xcor[k] = k/(float(zerl)-1)*360
        ycor[k] = Pa_angle[k]
        alph[k] = vinkel[i]
    
    
    cords=np.column_stack((xcor,alph,ycor))
    Pa_angle_x=np.concatenate((Pa_angle_x, cords))
    Pa_list.append((xcor,alph,ycor))
             

Pa_angle_x=Pa_angle_x[1:]

np.savetxt("Combined.txt", Pa_angle_x, delimiter=',')
np.save('Combined.npy', Pa_angle_x)

xxcut=int(np.min(zerlcount))

Pa_list0=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[0][0]) - 1, xxcut)).astype(int)
Pa_list0[0]= Pa_list[0][0][idx]
Pa_list0[1]= Pa_list[0][1][idx]
Pa_list0[2]= Pa_list[0][2][idx]

Pa_list1=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[1][0]) - 1, xxcut)).astype(int)
Pa_list1[0]= Pa_list[1][0][idx]
Pa_list1[1]= Pa_list[1][1][idx]
Pa_list1[2]= Pa_list[1][2][idx]

Pa_list2=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[2][0]) - 1, xxcut)).astype(int)
Pa_list2[0]= Pa_list[2][0][idx]
Pa_list2[1]= Pa_list[2][1][idx]
Pa_list2[2]= Pa_list[2][2][idx]

Pa_list3=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[3][0]) - 1, xxcut)).astype(int)
Pa_list3[0]= Pa_list[3][0][idx]
Pa_list3[1]= Pa_list[3][1][idx]
Pa_list3[2]= Pa_list[3][2][idx]

Pa_list4=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[4][0]) - 1, xxcut)).astype(int)
Pa_list4[0]= Pa_list[4][0][idx]
Pa_list4[1]= Pa_list[4][1][idx]
Pa_list4[2]= Pa_list[4][2][idx]

Pa_list5=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[5][0]) - 1, xxcut)).astype(int)
Pa_list5[0]= Pa_list[5][0][idx]
Pa_list5[1]= Pa_list[5][1][idx]
Pa_list5[2]= Pa_list[5][2][idx]

Pa_list6=np.zeros((3,xxcut))
idx = np.round(np.linspace(0, len(Pa_list[6][0]) - 1, xxcut)).astype(int)
Pa_list6[0]= Pa_list[6][0][idx]
Pa_list6[1]= Pa_list[6][1][idx]
Pa_list6[2]= Pa_list[6][2][idx]

P=np.concatenate((Pa_list0,Pa_list1,Pa_list2,Pa_list3,Pa_list4,Pa_list5,Pa_list6),axis=1)

P=np.transpose(P)

np.savetxt("Combined_equal_array_lengths.txt", P, delimiter=',')
np.save('Combined_equal_array_lengths.npy', P)

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
#white colormap
cm_bwr = make_colormap(
    [c('darkblue'), c('blue'), 0.2, c('blue'), c('mediumslateblue'), 0.4, c('mediumslateblue'), c('white'),0.475, c('white'),0.525, c('white'), c('salmon'), 0.6, c('salmon'), c('red'), 0.8, c('red'), c('darkred')])
#jet colormap
cm_bgr = make_colormap(
    [c('darkblue'), c('blue'), 0.2, c('blue'), c('cyan'), 0.4, c('cyan'), c('lightgreen'),0.49, c('lightgreen'),0.51, c('lightgreen'), c('yellow'), 0.6, c('yellow'), c('red'), 0.8, c('red'), c('darkred')])


