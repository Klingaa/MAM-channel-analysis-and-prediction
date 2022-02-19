import numpy as np
import math

input_list=np.load("output_data/input_list.npy")
name = input_list[0]
ps = float(input_list[1])
start =  0
end = int(input_list[3])-1

comp_through2 = np.load('Output_data/througharea_'+str(name)+'.npy')
dist_min_all = np.load('Output_data/dist_'+str(name)+'.npy')
cnt_area = np.load('Output_data/area_'+str(name)+'.npy')
surf_area = np.load('Output_data/surface_area_'+str(name)+'.npy')
volume = np.load('Output_data/inner_volume_'+str(name)+'.npy')

num_zeros = (comp_through2 == 0).sum()
num_ones = (comp_through2 == 1).sum()

########## DATA ANALYSIS FOR AREAL ##############
print ("The standard unit is micrometer")

#Length of investigated section
t_length = ps*(end-start+1)
print ("Length of the section is", t_length)

#Convert the pixelated distances to real values
dma_mu = dist_min_all*ps
print ("Resolution is", ps)

#Compute the maximum overall distance
lvmu = min(dma_mu)
print ("Largest valley (approx in mu)" , lvmu)

#Compute the maximum overall distance
lpmu = max(dma_mu)
print ("Largest peak (approx in mu)" , lpmu)

#Compute areal roughness value Sa
sa = np.mean(abs(dma_mu))
print ("Sa value is" , sa)

#Find 10 minimum and 10 maximum values
findx1 = np.sort(dma_mu)[-10:]
findx2 = np.sort(dma_mu)[0:10]
#Compute S10Z 
s10z = (abs(np.sum(findx1-findx2)))/10
print ("S10z value is" , s10z)

#Compute Sq
sq = np.sqrt(np.sum(np.power(dma_mu,2))/len(dma_mu))
print ("Sq value is" , sq)

#Compute skewness Ssk
ssk = np.sum(np.power(dma_mu,3))/(len(dma_mu)*sq**3)
print ("Ssk value is" , ssk)

#Compute kurtosis Sku
sku = np.sum(np.power(dma_mu,4))/(len(dma_mu)*sq**4)
print ("Sku value is" , sku)

#Compute the cross-sectional area using in-built area function
cnt_area=np.asarray(cnt_area)*ps**2
area_avg = np.mean(cnt_area)
print ("The average cross-sectional area is", area_avg)

#Compute the standard deviation of the through length areas
area_std = np.sqrt( np.sum((cnt_area-area_avg)**2)/(len(cnt_area)-1))
print ("The average cross-sectional area has STD", area_std)

#Compute average diameter based on the computed average cross-sectional area using in-built area function
Dh_A = 2*np.sqrt((area_avg)/math.pi)
print ("The average diameter is", Dh_A)

#Compute the standard deviation of the diameters computed from the areas of each slice
Dh_A_std = np.sqrt( np.sum((2*np.sqrt((cnt_area)/math.pi)-Dh_A)**2)/(len(cnt_area)-1))
print ("The average diameter has std", Dh_A_std)

through_area = num_ones*ps**2
print ('The through area is', through_area)

through_diameter=2*np.sqrt((through_area)/math.pi)
print ('The equivalent through diameter is', through_diameter)

#Compute the internal void volume from the overall avearge area and the number of slices
Internal_volume = area_avg/(end-start+1)*(end-start+1)*ps
print ('The internal void volume is', Internal_volume)

#Other method for computing the surface area by suing in-built function
SurfaceArea = surf_area
print ('The internal surface area is', SurfaceArea)

#Other method for computing the inner volume by counting number of inner pixels
TotVolOfInner = volume
print ('The internal void volume is', volume)

#Compute relation between area and volume
SSA2 = SurfaceArea/TotVolOfInner # Calculate specific surface area
print ('The specific surface area with is', SSA2)

#Save all computed results in txt file
file = open('Output_txt/Surface_deviation_analysis'+str(name)+'_'+str(ps)+'.txt',"w+")

file.write("_____The standard unit is micrometer_____\n\n")
file.write("Length of the investigated section is "+str(t_length)+' mu\n')
file.write("Resolution is "+str(ps)+' mu\n\n')
file.write("Largest valley is "+str(lvmu)+' mu\n')
file.write("Largest peak is "+str(lpmu)+' mu\n')
file.write("Sa value is "+str(sa)+' mu\n')
file.write("S10z value is "+str(s10z)+' mu\n')
file.write("Sq value is "+str(sq)+' mu\n')
file.write("Ssk value is "+str(ssk)+'\n')
file.write("Sku value is "+str(sku)+'\n\n')
file.write("The average cross-sectional area is "+ str(area_avg)+" mu^2\n")
file.write("The average cross-sectional area has std "+ str(area_std)+" mu^2\n")
file.write('The average diameter is '+str(Dh_A)+' mu\n')
file.write('The average diameter has std '+str(Dh_A_std)+' mu\n')
file.write('The through area is '+str(through_area)+' mu^2\n')
file.write('The equivalent through diameter is '+str(through_diameter)+' mu\n')
file.write('The internal surface area is '+str(SurfaceArea)+' mu^2\n')
file.write('The internal void volume is '+str(TotVolOfInner)+' mu^3\n')
file.write('The specific surface area with is '+str(SSA2)+' mu^-1\n')

file.close() 