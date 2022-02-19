import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
from matplotlib.pyplot import gca
plt.ion()

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])

#Defining the slice range
start =  0
end = int(input_list[3])-1

#Plotting font
afont = {'fontname':'Arial'}

centers0 = np.load('Output_data/centers0_'+str(name)+'.npy')
pp = np.load('Output_data/pp_'+str(name)+'.npy')
clist = np.load('Output_data/clist_'+str(name)+'.npy')
through = np.load('Output_data/througharea_'+str(name)+'.npy')
dist = np.load('Output_data/dist_'+str(name)+'.npy')
verts = np.load('Output_data/verts_'+str(name)+'.npy')
verts_e = np.load('Output_data/verts_e_'+str(name)+'.npy')
komp = np.load('Output_data/profiles_'+str(name)+'.npy')

#Creating of custum color schemes
def make_colormap(seq):
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

#Conversion of the color scheme to RGB
c = mcolors.ColorConverter().to_rgb

#Blue, White, Red colormap
cm_bwr = make_colormap(
    [c('darkblue'), c('blue'), 0.2, c('blue'), c('mediumslateblue'), 0.4, c('mediumslateblue'), c('white'),0.49, c('white'),0.51, c('white'), c('salmon'), 0.6, c('salmon'), c('red'), 0.8, c('red'), c('darkred')])

#Blue, Green, Red colormap
cm_bgr = make_colormap(
    [c('darkblue'), c('blue'), 0.2, c('blue'), c('cyan'), 0.4, c('cyan'), c('lightgreen'),0.49, c('lightgreen'),0.51, c('lightgreen'), c('yellow'), 0.6, c('yellow'), c('red'), 0.8, c('red'), c('darkred')])


#Channel profiles

ang1=0
ang2=360
zerl = len(komp[:,0,0])

fig4 = plt.figure(figsize=(10, 10))
ax4 = fig4.add_subplot(111, projection='3d')
for k in range(int((float(zerl)/360)*ang1),int((float(zerl)/360)*ang2)):
    sc4 = ax4.scatter(komp[k,:,0]*ps/1000,komp[k,:,1]*ps/1000,komp[k,:,2]*ps/1000, c=komp[k,:,3]*ps, cmap=cm_bgr,vmin=-1*math.ceil(max(dist)*ps), vmax=math.ceil(max(dist)*ps), s=10)
cbar4 = fig4.colorbar(sc4, shrink = 0.7)
cbar4.set_label('Distance to fitted \n surface [$\mu$m]',size=20,labelpad=15,**afont)
cbar4.ax.tick_params(labelsize=18) 
ax4.set_xlabel('x [mm]', fontsize=18,labelpad=15,**afont)
ax4.set_ylabel('y [mm]', fontsize=18,labelpad=15,**afont)
ax4.set_zlabel('z [mm]',fontsize=18,labelpad=15,**afont)
ax4.tick_params(labelsize=18)
ax4.set_zlim3d(0, len(clist[:,0])*ps/1000)
a4 = gca()
a4.set_xticklabels(a4.get_xticks(), **afont)
a4.set_yticklabels(a4.get_yticks(), **afont)
plt.show()

plt.savefig('Output_plots/Profiles_'+str(name)+'.tiff')