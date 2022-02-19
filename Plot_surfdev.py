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

dist = np.load('Output_data/dist_'+str(name)+'.npy')
verts = np.load('Output_data/verts_'+str(name)+'.npy')
verts_e = np.load('Output_data/verts_e_'+str(name)+'.npy')

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


#Channel surface points with deviation colors
fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111, projection='3d')
sc1 = ax1.scatter(verts[:,0]*ps/1000,verts[:,1]*ps/1000,verts[:,2]*ps/1000,c=dist*ps,cmap=cm_bgr,vmin=-1*math.ceil(max(dist)*ps), vmax=math.ceil(max(dist)*ps), s=1)#plt.cm.coolwarm
ax1.set_xlabel('x [mm]', labelpad=15, fontsize=18,**afont)
ax1.set_ylabel('y [mm]', labelpad=15, fontsize=18,**afont)
ax1.set_zlabel('z [mm]', labelpad=15, fontsize=18,**afont)
ax1.tick_params(labelsize=18)
cbar1 = fig1.colorbar(sc1, shrink = 0.7)
cbar1.set_label('Distance to fitted \n surface [$\mu$m]',size=20, labelpad=15,**afont)
cbar1.ax.tick_params(labelsize=18)
ax1.set_zlim3d(0, (end+1)*ps/1000)
a1 = gca()
a1.set_xticklabels(a1.get_xticks(), **afont)
a1.set_yticklabels(a1.get_yticks(), **afont)

#Save the plot
plt.savefig('Output_plots/Surfdevi_'+str(name)+'_fit_'+str(ps)+'.tiff')

