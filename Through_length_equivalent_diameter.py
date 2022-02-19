from __future__ import division
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import measure
import progressbar
from scipy import stats
from scipy.optimize import curve_fit
from skimage import img_as_ubyte
plt.ion()

input_list=np.load("output_data/input_list.npy")

#Defining the current name
name = input_list[0]

#Voxel size and channel size
ps = float(input_list[1])

#Defining the slice range
start =  0
end = int(input_list[3])-1

b_len=math. ceil(math.log(end/2)/math.log(2))
broek = np.zeros((b_len+1)).astype("int")
broek[0] = 1
for i in range(1,b_len+1):
    broek[i] = int(2**(i))

output=np.zeros((len(broek),np.max(broek)))

im_org = cv.imread('Image_slices/'+str(name)+''+'{0:04d}'.format(0)+'.tif',cv.CV_32S)
r,c = im_org.shape

SegTresh = 128
for k in range(0,len(broek)):
    j=broek[k]
    for ii in progressbar.progressbar(range(0,j)):
        nr_low =  int(0+int(input_list[3])/float(j)*ii)
        nr_high = int(0+int(input_list[3])/float(j)*ii+int(input_list[3])/float(j)-1)
        
        #Variable for stack of images
        im_stack = np.empty([r,c,nr_high-nr_low+1])
        
        #Variable for stack of images with isolated inner
        im_closed1 = np.zeros([r,c,nr_high-nr_low+1])
        comp_through = np.zeros([r,c])
        
        ########## ANALYSIS ##########
        
        #Read and create stack of images
        for i in range(nr_low,nr_high+1):
            im_temp1 = cv.imread('Image_slices/'+str(name)+''+'{0:04d}'.format(i)+'.tif',cv.CV_64F)
            im_temp11 = np.around((im_temp1-np.min(im_temp1))/float(np.max(im_temp1)-np.min(im_temp1))*65535).astype('uint16')
            im_temp2 = img_as_ubyte(im_temp11, force_copy=False)
            im_temp3 = cv.threshold(im_temp2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            im_stack[:,:,i-nr_low] = im_temp3[1]
        
        #Segment all images in the stack
        labelled_im_stack = measure.label((im_stack<SegTresh).astype('uint8')) # Labelling image, after segmentating volume
        
        #Define a kernel for dilation and erosion
        k_size1 = 1
        kernel1 = np.ones((k_size1,k_size1),np.uint8)
        
        
        for i in range(nr_low,nr_high+1):
            M = cv.moments(im_stack[:,:,i-nr_low])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            im = (labelled_im_stack[:,:,i-nr_low]==labelled_im_stack[cY,cX,i-nr_low]).astype('uint8')
            im_closed1[:,:,i-nr_low] = cv.morphologyEx(im, cv.MORPH_CLOSE, kernel1).astype('uint8')
        
           
        ##### THROUGH VIEW AREA #####
        comp_through[:,:] = 1
        for kk in range(0,len(im_closed1[0,0,:])):
            lok=np.where(im_closed1[:,:,kk]==0)
            comp_through[lok]=0
        
        comp_through2 = comp_through
        
        num_zeros = (comp_through2 == 0).sum()
        num_ones = (comp_through2 == 1).sum()
        
        through_area = num_ones*ps**2
        
        through_diameter=2*np.sqrt((through_area)/math.pi)
        
        output[k,ii]=through_diameter
        
        #Plotting font
        afont = {'fontname':'Arial'}
    
    output[output == 0] = np.nan
    print (np.nanmean(output[k,:]))
    
yy = np.nanmean(output[:,:],axis=1)
print (np.nanmean(output[:,:],axis=1))

#fig = plt.figure(figsize=(16, 9))
#ax = fig.add_subplot(111)
#sc = ax.scatter(1/np.asarray(broek),yy, c = 'k', s = 50)
#ax.grid(True)
#ax.set_ylabel('Equivalent through diameter [$\mu m$]', labelpad=14, fontsize=20)
#ax.set_xlabel('Relative batch size [-]', labelpad=14, fontsize=20)
#ax.tick_params(labelsize=20)
#ax.xaxis.label.set_color('black')
#ax.yaxis.label.set_color('black')
#plt.show()
#plt.savefig('Output_plots/'+str(name)+'_through_diameter_batch.tiff')

lengde = ps*int(end)/float(1000)


xx=lengde/broek

xx_pred = np.arange(lengde/float(broek[-1]),lengde/float(0.25/4),1/float(100))
xx_pred = xx_pred[::-1]


fitz1 = np.polyfit(xx, yy, 1)
p1 = np.poly1d(fitz1)
pp1 = np.zeros((len(xx_pred),2))
pp1[:,0] = xx_pred
pp1[:,1] = p1(xx_pred)

fitz2 = np.polyfit(xx, yy, 2)
p2 = np.poly1d(fitz2)
pp2 = np.zeros((len(xx_pred),2))
pp2[:,0] = xx_pred
pp2[:,1] = p2(xx_pred)

fitz3 = np.polyfit(xx, yy, 3)
p3 = np.poly1d(fitz3)
pp3 = np.zeros((len(xx_pred),2))
pp3[:,0] = xx_pred
pp3[:,1] = p3(xx_pred)

weights = np.zeros((len(broek))).astype("int")
weights[-1] = 1
for i in range(1,b_len+1):
    weights[-1-i] = 1*10**i

def logfunc(x, a, b):
    return a*np.log(x)+b
fitz4 = np.polyfit(np.log(xx), yy, 1,w=weights)
pp4 = np.zeros((len(xx_pred),2))
pp4[:,0] = xx_pred
pp4[:,1] = logfunc(xx_pred,*fitz4)

err = np.mean(logfunc(xx,*fitz4)-yy)
errscale = err/float(np.max(xx)-np.min(xx))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
sc = ax.scatter(xx,yy, c = 'k', s = 50)
sc1 = ax.plot(xx,p1(xx), c = 'r')
sc2 = ax.plot(xx,p2(xx), c = 'b')
sc3 = ax.plot(xx,p3(xx), c = 'y')
sc4 = ax.plot(xx,logfunc(xx,*fitz4),c = 'm')
ax.grid(True)
ax.set_xlabel('Length [mm]', labelpad=14, fontsize=20)
ax.set_ylabel('Equivalent through diameter [$\mu m$]', labelpad=14, fontsize=20)
ax.tick_params(labelsize=20)
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
plt.legend([sc,sc1[0],sc2[0],sc3[0],sc4[0]], ['Data', 'y = '+str(np.round(fitz1[0],3))+' * x + '+str(np.round(fitz1[1],3)), \
'y = '+str(np.round(fitz2[0],3))+' * x$^2$ + '+str(np.round(fitz2[1],3))+' * x + '+str(np.round(fitz2[2],3)), \
'y = '+str(np.round(fitz3[0],3))+' * x$^3$ + '+str(np.round(fitz3[1],3))+' * x$^2$ + '+str(np.round(fitz3[2],3))+' * x + '+str(np.round(fitz3[3],3)), \
'y = '+str(np.round(fitz4[0],3))+' * ln(x) + '+str(np.round(fitz4[1],3))],fontsize = 20,loc='upper right')
plt.show()
plt.savefig('Output_plots/'+str(name)+'_through_diameter_length_fits.tiff')

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
sc = ax.scatter(xx,yy, c = 'k', s = 50)
sc4 = ax.plot(pp4[:,0],pp4[:,1],c = 'm')
ax.grid(True)
ax.set_xlabel('Length [mm]', labelpad=14, fontsize=20)
ax.set_ylabel('Equivalent through diameter [$\mu m$]', labelpad=14, fontsize=20)
# ax.set_xlim(0, 50)
ax.tick_params(labelsize=20)
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
plt.legend([sc,sc4[0]], ['Data', 'y = '+str(np.round(fitz4[0],3))+' * ln(x) + '+str(np.round(fitz4[1],3))],fontsize = 20,loc='upper right')
plt.show()
plt.savefig('Output_plots/'+str(name)+'_through_diameter_length_ln.tiff')