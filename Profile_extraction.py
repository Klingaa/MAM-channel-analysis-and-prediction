import numpy as np
import progressbar

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])
D = float(input_list[2])

#Defining the slice range
start =  0
end = int(input_list[3])-1

from Custom_Functions import CutAng
from Custom_Functions import GetAngle

#Define the base cutoff angle
cutref = CutAng(D,ps)

#Normalize the range for computations

#Loading deviation data 
dist = np.load('Output_data/dist_'+str(name)+'.npy')

#Loading corresponding point cloud from P1
verts = np.load('Output_data/verts_'+str(name)+'.npy')

#Loading ellipse centers
cnt_ellipse = np.load('Output_data/e_centers_'+str(name)+'.npy')

#Save the unedited deviations
dma_mu = dist

#Save another instance of the deviations
dma_hist = dma_mu

#Define a matric for a combination of the point cloud data and deviations
vnew = np.zeros((len(dist),4))

#Input the point cloud data in the new matrix
vnew[:,0:3] = verts[:,0:3]

#Input the corresponding deviations in the new matrix
vnew[:,3] = dist

#Sort the new matrix from low to high with respect to the z-values
vnew = vnew[vnew[:,2].argsort()]

#Round all z-values
vnew[:,2] = np.round(vnew[:,2])

########## ANALYSIS ##########

#Determine where the z-values are 0 in order to identify all "seeds"
zer = np.where(vnew[:,2]==0)

#Convert to array
zer = np.array(zer)

#Compute the center of the data in the lowest slice
x0 = cnt_ellipse[0,0]
y0 = cnt_ellipse[0,1]

#Define a list for the angles
vinkellist = []

#Compute angle between center of slice z = 0 and all seeds in that slice
for g in range(0,len(vnew[zer[0],0])):
    vinkel = GetAngle(vnew[g,0:2],[x0,y0])
    vinkellist.append(vinkel)
vinkellist = np.array(vinkellist)
vinkelcomb = np.zeros((len(zer[0]),2))
vinkelcomb[:,0] = vinkellist
vinkelcomb[:,1] = zer[0]
#Sort the seeds with respect to the angle, from -180 to 180
vinkelcomb = vinkelcomb[vinkelcomb[:,0].argsort()]

#The number of seeds
zerl = len(zer[0,:])

#Define variables for computations
komp = []
refkomp = []
vnewold = []
center = []

#Define the possibly changing cutoff angle
cutoff = cutref

#Define all centers of the slices from
centers0 = []

for k in progressbar.progressbar(range(start,end)):
        x01 = cnt_ellipse[k+1,0]
        y01 = cnt_ellipse[k+1,1]
        centers0.append([x01,y01,k+1])
        
#Convert to array
centers0 = np.array(centers0)

#Fit a polynomium to the slice centers for describing the direction of the channel
fitzx = np.polyfit(centers0[:,2], centers0[:,0], 1)
fitzy = np.polyfit(centers0[:,2], centers0[:,1], 1)

#Convert it to a 1D fit
p1 = np.poly1d(fitzx)
p2 = np.poly1d(fitzy)

#Create a matric for all the centers of the 1D fit
pp = np.zeros((len(centers0),3))

#Compute the coordinates
pp[:,0] = p1(centers0[:,2])
pp[:,1] = p2(centers0[:,2])
pp[:,2] = centers0[:,2]

#Create the direction vector
pp_vec=pp[-1]-pp[0]
pp_vec=pp_vec/pp_vec[2]

#Generation of profiles from one seed at a time
for j in progressbar.progressbar(range(0,zerl)):
    #The index of the first seed
    inp =  int(vinkelcomb[j,1]) #j
    #The coordinates of the first seed
    cp = vnew[inp]
    #A list of the center coordinates
    clist = []
    #The seed will act as the first point in the list, after which the new found points will be added later
    clist.append(cp)
    #Saving the current point for reference
    vnewold = []
    vnewold = vnew[inp]
    #Another instance of saving for reference
    refplot = []
    refplot.append(cp[0:3])
    #Start looking for the next point in the next slice for the profile
    for i in range(start+1,end):
        #Determine which points that are in the next slice
        lok = np.where(vnew[:,2]==i)
        #Convert to array
        lok = np.array(lok)
        #Len of that array equals the number of points
        l = len(lok[0])
        #Determine the center points of that slice
        x = pp[i,0]
        y = pp[i,1]
        #Determine the coordinates of the prior point in the profile
        run = clist[i-1]
        #Compute the distance from the former point to the new point
        dic = vnew[lok[0,:],3]
        #Save the distance
        dic2=dic
        #Define list for testing the angle    
        testlist = []
        #Compute the differences in the angles between all points in that slice and the point from the former slice with respect to the center point of the current slice
        for ii in range(0,len(dic)):
            ang1 = GetAngle([x,y],[vnew[lok[0,ii],0],vnew[lok[0,ii],1]])
            ang2 = GetAngle([x,y],[vnewold[0],vnewold[1]])
            diff = abs(ang1-ang2)
            testlist.append(diff)
        testlist=np.asarray(testlist)
        #Find where the differences is below the cutoff
        lok2 = np.where(testlist < cutoff)
        #Determine the maximum distance from the from the points that fulfilled the angular criterion
        fi = np.argmax(dic[lok2])
        lok3 = np.asarray(lok2)
        fi = lok3[0,fi]
        #add the new point to the list of points for the profile
        clist.append(vnew[lok[0,fi],:])
        #Update the optimum path of the profile with the direction vector
        vnewold = vnewold[0:3]+pp_vec
        #Reset cutoff value
        cutoff = cutref
    #Save the coordinates of the point
    cord=np.array(clist)
    #Save the complete point for the profile in the complete list
    komp.append(cord)
#Convert to array
komp=np.array(komp)
clist=np.array(clist)

#Save data for later use in other codes
np.save("Output_data/"+'profiles_'+str(name)+'.npy', komp[:,1:,:])
np.save("Output_data/"+'centers0_'+str(name)+'.npy', centers0[1:,:])
np.save("Output_data/"+'pp_'+str(name)+'.npy', pp[1:,:])
np.save("Output_data/"+'clist_'+str(name)+'.npy', clist[1:,:])

