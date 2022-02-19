import cv2 as cv
import numpy as np
from skimage import measure
import progressbar
from skimage import img_as_ubyte

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size
ps = float(input_list[1])

#Defining the slice range
start =  0
end = int(input_list[3])-1

#Sizing up the images
im_org = cv.imread('Image_slices/'+str(name)+''+'{0:04d}'.format(0)+'.tif',cv.CV_64F)
r,c = im_org.shape

#Variable for stack of images
labelled_im_stack = np.empty([r,c,end-start+1])
im_stack = np.empty([r,c,end-start+1])

#Variable for stack of images with isolated inner
im_closed1 = np.zeros([r,c,end-start+1])
comp_through = np.zeros([r,c])

#Variable for stack of ellipses
im_ellipse_stack = np.zeros([r,c,end-start+1])

#Variable for stack of contours
cnt_stack = []

#Variable for stack of fitted ellipses
cnt_ellipse_stack = []

#Variable for stack of ellipse centers
ellipse_centroids = []

########## ANALYSIS ##########
SegTresh = 128
#Read and create stack of images
for i in progressbar.progressbar(range(start,end+1)):
    im_temp1 = cv.imread('Image_slices/'+str(name)+''+'{0:04d}'.format(i)+'.tif',cv.CV_64F)
    im_temp11 = np.around((im_temp1-np.min(im_temp1))/float(np.max(im_temp1)-np.min(im_temp1))*65535).astype('uint16')
    im_temp2 = img_as_ubyte(im_temp11, force_copy=False)
    im_temp3 = cv.threshold(im_temp2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    im_stack[:,:,i-start] = im_temp3[1]

#Segment all images in the stack
labelled_im_stack = measure.label((im_stack<SegTresh).astype('uint8')) # Labelling image, after segmentating volume

#Define ininial length of the contour
cnt_length = 0

#Define the initial inner area of the contour
cnt_area = []

#Variable for listing the lengths of contours
cnt_list = []

#Define a kernel for dilation and erosion
k_size1 = 1
k_size2 = 25
kernel1 = np.ones((k_size1,k_size1),np.uint8)
kernel2 = np.ones((k_size2,k_size2),np.uint8)

#Fitting an ellipse to each slice and recording relevant data
for i in progressbar.progressbar(range(start,end+1)):
    #Determine the center
    M = cv.moments(im_stack[:,:,i-start])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #Use the center to isolate inner void
    im = (labelled_im_stack[:,:,i-start]==labelled_im_stack[cY,cX,i-start]).astype('uint8')
    #Perform dilation and erosion
    im_closed1[:,:,i-start] = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel1).astype('uint8')
    im_closed2 = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel2).astype('uint8')
    im_closed3 = im_closed2.astype('uint8')
    #Define an empty image
    im_t = np.zeros([r,c])
    #Find correct contour
    contours = cv.findContours(im_closed3,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)[0]
    #Log the correct contour (here it is the largest thus "-1")
    ka = np.zeros((len(contours),1))
    for i1 in range(0,len(ka)):
        ka[i1] = len(contours[:][i1])
        
    kag = max(ka)
    hh = np.where((kag==ka))[0][0]
    
    cnt = contours[hh]
    #Save the contour in the stack of contours
    cnt_stack.append(contours)
    #Fit an ellipse to the contour
    ellipse = cv.fitEllipse(cnt) # Fit ellipse to contour coordinates
    #Draw the fitted ellipse in the empty image
    cv.ellipse(im_t,ellipse,(1,0,0),-1)
    #Save the drawn ellipse in the stack of fitted ellipses
    im_ellipse_stack[:,:,i-start] = im_t
    #Save the center of the ellipse
    ellipse_centroids.append(ellipse[0])
    #Compute the accumulated length of the contours
    cnt_length = cnt_length + cv.arcLength(cnt,True)
    #Save the accumulated length
    cnt_list.append(cnt_length)
    #Define the contour of the ellipse
    contours_e = cv.findContours(im_t.astype('uint8'),cv.RETR_TREE,cv.CHAIN_APPROX_NONE)[0]
    #Save the area of the contour
    cnt_area.append(abs(cv.contourArea(cnt)))
    #Save the contour to the stack of ellipse contours
    cnt_ellipse_stack.append(contours_e[0])
	
# Surface is approximated using Lewiner's marching cubes
verts, faces,_,_ = measure.marching_cubes_lewiner((labelled_im_stack==labelled_im_stack[cY,cX,end-start]).astype('uint8'),level=0.999)

#Define an empty array for center coordinates for ellipses
cnt_coords = np.empty([1,3])

#Create z-coordinates for 
for i in progressbar.progressbar(range(start,end+1)):
    #Find center of ellipse
    c = cnt_ellipse_stack[i-start][:,0,:]
    #Find the z-coordinate
    cc = np.c_[c,np.ones(len(c))*(i-start)]
    #Add the z-coordinate to the center coordinates for the ellipse
    cnt_coords = np.r_[cnt_coords,cc]
    
# Variable for minimum distance between points on inner surface and surface defined as inner ellipse
dist_min_all = np.empty(len(verts))

#Variable for index of the minimum distance
dist_minIdx_all = np.empty(len(verts))	

#Create surface mesh of the fitted ellipses stack
verts_e, faces_e,_,_ = measure.marching_cubes_lewiner(im_ellipse_stack,level=0.999) 

#Compute the minimum distance and find its index by running through all points in channel mesh
for i in progressbar.progressbar(range(0,len(verts))):
    #Define a point
    p = verts[i,:]
    #Compute its distance to all points on surface of fitted ellipses
    d = np.sqrt((p[0]-verts_e[:,0])**2+(p[1]-verts_e[:,1])**2+(p[2]-verts_e[:,2])**2)
    #Define the direction vector from the point to the minimum distance point on fitted ellipses mesh
    distvec = [verts_e[np.argmin(d),0]-p[0],verts_e[np.argmin(d),1]-p[1],verts_e[np.argmin(d),2]-p[2]]
    #Load the center of the ellipse at the z-coordinate of the minimum distance point
    centroid = ellipse_centroids[verts_e[np.argmin(d),2].astype('uint8')] 
    #Define the direction vector from the center of the ellipse to the analyzed point
    centroidvec = p-[centroid[1],centroid[0],verts_e[np.argmin(d),2]]
    #Add the sign to the distance of the point by using the scalar product of the two direction vectors
    dist_min_all[i] = np.sign(np.dot(distvec,centroidvec))*np.min(d)
    #Compute the index of the minimum distance point on the fitted ellipse surface
    dist_minIdx_all[i] = np.argmin(d)

##### THROUGH VIEW AREA #####
comp_through[:,:] = 1
for kk in range(0,len(im_closed1[0,0,:])):
    lok=np.where(im_closed1[:,:,kk]==0)
    comp_through[lok]=0

comp_through2 = comp_through
np.save('Output_data/'+'througharea_'+str(name)+'.npy', comp_through2)
#Convert the pixelated distances to real values
dma_mu = dist_min_all*ps
#Other method for computing the surface area by suing in-built function
SurfaceArea = ps**2*measure.mesh_surface_area(verts,faces)
#Other method for computing the inner volume by counting number of inner pixels
TotVolOfInner = ps**3*np.sum(np.sum(np.sum((labelled_im_stack==labelled_im_stack[cY,cX,end-start]).astype('uint8')))) # Calculate volume contained by surface as the number of voxels

#Save data for later use in other codes
np.save('Output_data/'+'verts_'+str(name)+'.npy', verts)
np.save('Output_data/'+'verts_e_'+str(name)+'.npy', verts_e)
np.save('Output_data/'+'dist_'+str(name)+'.npy', dist_min_all)
np.save('Output_data/'+'faces_'+str(name)+'.npy', faces)
np.save('Output_data/'+'faces_e_'+str(name)+'.npy', faces_e)
np.save('Output_data/'+'area_'+str(name)+'.npy', cnt_area)
np.save('Output_data/'+'e_centers_'+str(name)+'.npy', ellipse_centroids)
np.save('Output_data/'+'e_centers_'+str(name)+'.npy', ellipse_centroids)
np.save('Output_data/'+'surface_area_'+str(name)+'.npy', SurfaceArea)
np.save('Output_data/'+'inner_volume_'+str(name)+'.npy', TotVolOfInner)

#save xyzd array
xyzd_name=str("x y z d")
xyzd = np.concatenate((verts,np.expand_dims(dma_mu, axis=1)),axis=1)
np.savetxt('Output_txt/XYZD_'+str(name)+'_'+str(ps)+'.txt', xyzd, header=xyzd_name, comments='')