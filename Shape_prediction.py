#author: cgkli@mek.dtu.dk

########## PRE ANALYSIS ##########

#Loading relevant packages
import matplotlib.pyplot as plt
import numpy as np
import math
import progressbar
import matplotlib.colors as mcolors
from stl import mesh
from math import atan2,degrees
import open3d as o3d
plt.ion()

#Defining the current name
name = "CCC"

#Load the STL/PCloud
Ra_max0 = 4
#Input the model

a = 20
b= 180

#p00f =      -0.4322  
#p10f =      -0.7049  
#p01f =      0.3833  
#p20f =      0.05308  
#p11f =      0.01322  
#p02f =      -0.001074  
#p30f =       -0.001525  
#p21f =     -0.0004823  
#p12f =       -3.83e-05  
#p40f =    1.99e-05  
#p31f =    2.922e-06  
#p22f =       1.438e-06  
#p50f =       -9.378e-08  
#p41f =      3.229e-09  
#p32f =      -9.801e-09  
#
#def full_model(a,b):
#    return p00f + p10f*a + p01f*b + p20f*a**2 + p11f*a*b + p02f*b**2 + p30f*a**3 + p21f*a**2*b \
#    + p12f*a*b**2 + p40f*a**4 + p31f*a**3*b + p22f*a**2*b**2 + p50f*a**5 \
#    + p41f*a**4*b + p32f*a**3*b**2
#
#print full_model

p00s1 =       13.81 
p10s1 =     -0.0423 
p01s1 =    -0.04385  
p20s1 =  -0.0002997  
p11s1 =   0.0004172  
p02s1 =  -0.0003772  


def side1_model(a,b):
    return p00s1 + p10s1*a + p01s1*b + p20s1*a**2 + p11s1*a*b + p02s1*b**2

p00s2 =       78.04  
p10s2 =      0.3699  
p01s2 =     -0.5084  
p20s2 =  -0.0004752  
p11s2 =   -0.001096  
p02s2 =   0.0009163  

def side2_model(a,b):
    return p00s2 + p10s2*a + p01s2*b + p20s2*a**2 + p11s2*a*b + p02s2*b**2

p00p =      -64.64
p10p =       -1.44
p01p =       1.187
p20p =     0.08176
p11p =     0.02498
p02p =   -0.003284
p30p =   -0.001795
p21p =  -0.0008641
p12p =  -8.003e-05
p40p =   2.454e-05
p31p =   3.032e-06
p22p =    2.99e-06
p50p =  -1.283e-07
p41p =   1.976e-08
p32p =   -1.86e-08

def peak_model(a,b):
    return p00p + p10p*a + p01p*b + p20p*a**2 + p11p*a*b + p02p*b**2 + p30p*a**3 + p21p*a**2*b \
    + p12p*a*b**2 + p40p*a**4 + p31p*a**3*b + p22p*a**2*b**2 + p50p*a**5 \
    + p41p*a**4*b + p32p*a**3*b**2

peak_min = 70
peak_max = 290


def fitPlaneLTSQ(XYZ):
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  #X
        G[:, 1] = XYZ[:, 1]  #Y
        Z = XYZ[:, 2]
        (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
        normal = (a, b, -1)
        nn = np.linalg.norm(normal)
        normal = normal / nn
        return (c, normal)

def GetAngle(p1, p2):
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        zDiff = p2[2] - p1[2]
        return degrees(atan2(zDiff, yDiff, xDiff))
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

test_mesh = mesh.Mesh.from_file(str(name)+'.stl')

##########

model_select = 0 

Ra_factor = 1 #How heavily Ra should affect, 2 is standard 3 is maximum

mesh_normals=test_mesh.normals

test_points=test_mesh.points

comb=np.concatenate((test_points[:,0:3], test_points[:,3:6],test_points[:,6:9]), axis=0)

xyz = np.unique(comb, axis=0)



names = ['x','y','z','xcent','ycent','zcent','UpOrDownFacing(1or0)','Nx','Ny','Nz','alpha','beta','Pa']
# print names
#number of nearest points 
an = 30 # sampling size for normal determination
bn=int(len(xyz[:,0])*0.01) # sampling size for determination of upward or downward facing surface through center of mass

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.geometry.PointCloud.estimate_normals(pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000,max_nn=an))
o3d.geometry.PointCloud.orient_normals_to_align_with_direction(pcd)

#Define build direction
bd=np.asarray((0,0,1))

total_mat=np.zeros(([len(xyz[:,0]),14]))

total_mat[:,0:3]=xyz

total_mat[:,7:10]=pcd.normals#normal

f_near=np.zeros((len(xyz[:,0]),bn,3))

x = np.zeros(([len(xyz[:,0]),1]))


###
dotpd=np.zeros(([len(xyz[:,0]),1]))
norm=np.zeros(([len(xyz[:,0]),3]))

        
for i in progressbar.progressbar(range(0,len(xyz[:,0]))):#
    d = abs(np.sqrt((xyz[i,0]-xyz[:,0])**2+(xyz[i,1]-xyz[:,1])**2+(xyz[i,2]-xyz[:,2])**2))
    f_dist=sorted(d)[0:bn]
    f_dist=np.asarray(f_dist)
    
    total_mat[:,13] = 0
    
    for k in range(0,bn):
        lok=np.where(d == f_dist[k])
        
        for h in range(0,len(lok[0][:])):
            if total_mat[lok[0][h],13] == 0:
                f_near[i,k,:] = xyz[lok[0][h]]
                total_mat[lok[0][h],13] = 1
                break
    
    centroid = np.asarray((np.mean(f_near[i,:,0]), np.mean(f_near[i,:,1]), np.mean(f_near[i,:,2])))
    centroid=np.reshape(centroid, (1, 3))
    total_mat[i,3:6]=centroid
    
    norm[i,0:3] = total_mat[i,7:10]
    centvec=total_mat[i,3:6]-total_mat[i,0:3]
    dotpd[i,0]=np.dot(centvec,total_mat[i,7:10])
    if dotpd[i] < 0:
        norm[i,0:3] = -1*total_mat[i,7:10]


    if norm[i,2] >= 0:
        total_mat[i,6] = 1
    
    total_mat[i,9]=abs(total_mat[i,9])
    angle_a = angle_between(total_mat[i,7:10],bd)*180/math.pi
    total_mat[i,10] = angle_a
    total_mat[i,11] = 180-angle_a
    
    if total_mat[i,6] == 1:
        total_mat[i,11] = angle_a
    
    a=total_mat[i,10]
    b=total_mat[i,11]
    
    
    if model_select == 0:
        if total_mat[i,11]<peak_min:
            total_mat[i,12] = side1_model(a,b)
        if peak_max<total_mat[i,11]:
            total_mat[i,12] = side2_model(a,b)
        if peak_min<=total_mat[i,11]<=peak_max:
            total_mat[i,12] = peak_model(a,b)
        if total_mat[i,12] < 5:
            total_mat[i,12] = 5

np.save('method1_alpha_beta.npy', total_mat)

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


#Blue, Green, Red colormap
cm_bgr = make_colormap(
    [c('darkblue'), c('blue'), 0.2, c('blue'), c('cyan'), 0.4, c('cyan'), c('lightgreen'),0.49, c('lightgreen'),0.51, c('lightgreen'), c('yellow'), 0.6, c('yellow'), c('red'), 0.8, c('red'), c('darkred')])

#Green, Red colormap
cm_gr = make_colormap(
    [c('lightgreen'), c('yellow'),0.25, c('yellow'), c('red'), 0.75, c('red'), c('darkred')])


maxdist=np.max(np.min([np.max(total_mat[:,0])-np.min(total_mat[:,0]),np.max(total_mat[:,1])-np.min(total_mat[:,1]),np.max(total_mat[:,2])-np.min(total_mat[:,2])]))

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(total_mat[:,0],total_mat[:,1],total_mat[:,2],total_mat[:,7],total_mat[:,8],total_mat[:,9],20)
# ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
# ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
# ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)
# ax.set_xlim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
# ax.set_ylim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
# ax.set_zlim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
 

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(total_mat[:,0],total_mat[:,1],total_mat[:,2],c=total_mat[:,10],cmap=cm_bgr,s=5,vmin=0,vmax=90)#plt.cm.coolwarm
ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)
cbar = fig.colorbar(sc, shrink = 0.7)
cbar.set_label('$\\alpha$ [$\degree$]',size=20, labelpad=15)
cbar.ax.tick_params(labelsize=15) 
ax.set_xlim3d((np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2-1.25*maxdist/2,(np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2+1.25*maxdist/2)
ax.set_ylim3d((np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2-1.25*maxdist/2,(np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2+1.25*maxdist/2)
ax.set_zlim3d((np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2-1.25*maxdist/2,(np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2+1.25*maxdist/2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(total_mat[:,0],total_mat[:,1],total_mat[:,2],c=total_mat[:,11],cmap=cm_bgr,s=5,vmin=0,vmax=180)#plt.cm.coolwarm
ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)
cbar = fig.colorbar(sc, shrink = 0.7)
cbar.set_label('$\\beta$ [$\degree$]',size=20, labelpad=15)
cbar.ax.tick_params(labelsize=15)
ax.set_xlim3d((np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2-1.25*maxdist/2,(np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2+1.25*maxdist/2)
ax.set_ylim3d((np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2-1.25*maxdist/2,(np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2+1.25*maxdist/2)
ax.set_zlim3d((np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2-1.25*maxdist/2,(np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2+1.25*maxdist/2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(total_mat[:,0],total_mat[:,1],total_mat[:,2],c=total_mat[:,12],cmap=cm_gr, s=5, vmin=0,vmax=np.max(total_mat[:,12]))#vmin=-np.max(total_mat[:,12]),vmax=np.max(total_mat[:,12]))
ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)
cbar = fig.colorbar(sc, shrink = 0.7)
cbar.set_label('Pa [$\mu m$]',size=20, labelpad=15)
cbar.ax.tick_params(labelsize=15)
ax.set_xlim3d((np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2-1.25*maxdist/2,(np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2+1.25*maxdist/2)
ax.set_ylim3d((np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2-1.25*maxdist/2,(np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2+1.25*maxdist/2)
ax.set_zlim3d((np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2-1.25*maxdist/2,(np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2+1.25*maxdist/2)

shape_cords=total_mat[:,0:3]
xx=np.random.uniform(low=-1,high=1,size=(len(total_mat[:,12])))
dotpd=np.zeros(([len(xyz[:,0]),1]))
norm=np.zeros(([len(xyz[:,0]),3]))

for ii in range(0,len(xyz)):
    norm[ii,0:3] = total_mat[ii,7:10]
    centvec=total_mat[ii,3:6]-total_mat[ii,0:3]
    dotpd[ii,0]=np.dot(centvec,total_mat[ii,7:10])
    if dotpd[ii] < 0:
        norm[ii,0:3] = -1*total_mat[ii,7:10]
        
    Ra_factor = 1
    
    Ra_max = Ra_max0
    
    if total_mat[ii,11]>90:
        Ra_factor=1*total_mat[ii,11]/float(90)+(Ra_max-1)/float(90)*(total_mat[ii,11]-90)
    
    shape_cords[ii,0:3]=shape_cords[ii,0:3]+norm[ii,0:3]*total_mat[ii,12]/1000*Ra_factor+xx[ii]*norm[ii,0:3]*total_mat[ii,12]/1000*Ra_factor/2


pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(shape_cords)
o3d.io.write_point_cloud("new_shape.ply", pcd2)
np.save('method1_alpha_beta_newcords.npy', shape_cords)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(shape_cords[:,0],shape_cords[:,1],shape_cords[:,2],c=total_mat[:,12],cmap=cm_gr, s=5, vmin=0,vmax=np.max(total_mat[:,12]))# vmin=-np.max(total_mat[:,12]),vmax=np.max(total_mat[:,12]))#plt.cm.coolwarm
ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)
cbar = fig.colorbar(sc, shrink = 0.7)
cbar.set_label('Pa [$\mu m$]',size=20, labelpad=15)
cbar.ax.tick_params(labelsize=15) 
ax.set_xlim3d((np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2-1.25*maxdist/2,(np.min(total_mat[:,0])+np.max(total_mat[:,0]))/2+1.25*maxdist/2)
ax.set_ylim3d((np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2-1.25*maxdist/2,(np.min(total_mat[:,1])+np.max(total_mat[:,1]))/2+1.25*maxdist/2)
ax.set_zlim3d((np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2-1.25*maxdist/2,(np.min(total_mat[:,2])+np.max(total_mat[:,2]))/2+1.25*maxdist/2)


pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(shape_cords)
o3d.io.write_point_cloud(str(name)+"_Ra_max"+str(Ra_max0)+".ply", pcd2)

ij = 1250
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(f_near[ij,1:,0],f_near[ij,1:,1],f_near[ij,1:,2], s=50,color='b')
sc1 = ax.scatter(f_near[ij,0,0],f_near[ij,0,1],f_near[ij,0,2], s=50,color='r')
sc2 = ax.scatter(total_mat[ij,3],total_mat[ij,4],total_mat[ij,5], s=50,color='k')
ax.set_xlabel('x [mm]', labelpad=14, fontsize=13)
ax.set_ylabel('y [mm]', labelpad=14, fontsize=13)
ax.set_zlabel('z [mm]', labelpad=6, fontsize=13)


xx=0

# fig2 = plt.figure(figsize=(10, 10))
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.quiver(xyz[:,0],xyz[:,1],xyz[:,2],total_mat[:,7]/1000,total_mat[:,8]/1000,total_mat[:,9]/1000,50)
# ax2.set_xlabel('x [mm]', labelpad=14, fontsize=13)
# ax2.set_ylabel('y [mm]', labelpad=14, fontsize=13)
# ax2.set_zlabel('z [mm]', labelpad=6, fontsize=13)
# ax2.set_xlim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
# ax2.set_ylim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
# ax2.set_zlim3d(np.min(total_mat[:,0:3]), np.max(total_mat[:,0:3]))
# ax2.view_init(0, -90)