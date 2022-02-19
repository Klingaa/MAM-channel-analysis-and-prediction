########## PRE ANALYSIS ##########

#Loading relevant packages
import numpy as np
import matplotlib.pyplot as plt
import progressbar
plt.ion()
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import open3d as o3d
#from open3d import *

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])

#Loading data for analysis from P1 and P2
verts = np.load('Output_data/verts_'+str(name)+'.npy')
dist = np.load('Output_data/dist_'+str(name)+'.npy')
#komp = np.load('Output_data/profiles_'+str(name)+'.npy')

#Printing maximum and minimum for helping determining color ranges
print ("\n")
print (max(dist)*ps)
print (min(dist)*ps)
print ("\n")

#Define color ranges
Manf1max = max(dist)#456/ps
Manf1min = min(dist)#-241/ps

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

#Define normalization
norm = Normalize(vmin=-Manf1max, vmax=Manf1max)

#Snit is used for partial views
snit= 0

#Locate the points for viewing
lok = np.where(verts[:,1] > snit)

#Update the plotted points
verts2 = verts[lok]

#Corresponding distance values for the points for plotting
dist2 = dist[lok]

#Normalize the points
dist2 = norm(dist2)

#Define the colormatrix
farve = np.zeros((len(dist2),3))

#Add the colors
for i in progressbar.progressbar(range (0,len(dist2))):
    farve[i,:] = cm_bgr(dist2[i])[0:3]

########## PLOTTING ##########

#Create a pointcloud for plotting
pcd = o3d.geometry.PointCloud()
#Add the points from the data
pcd.points = o3d.utility.Vector3dVector(verts2)
#Give the points color
pcd.colors = o3d.utility.Vector3dVector(farve)
#Temporarily save the pointcloud data
o3d.io.write_point_cloud("Output_data/sync.ply", pcd)
#Load the temporary data
pcd_load = o3d.io.read_point_cloud("Output_data/sync.ply")
#Draw and plot the data with color
o3d.visualization.draw_geometries([pcd_load])