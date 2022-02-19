import numpy as np

name = "s01_image_slices"
ps = 15.8304
D = 3000
stacksize = 500

input_list = np.array([name,ps,D,stacksize])
np.save("Output_data\input_list",input_list)
    
import sys
sys.path.insert(1, 'Codes')
    
import Surface_deviation

import Analysis_parameters

import Plot_surfdev #Plot_surfdev_down

import Plot_fitstack

import Plot_through
    
import Profile_extraction

import Plot_profiles 
 
import Analysis_parameters_profile

import Plot_profile_parameters
    
import Through_length_equivalent_diameter #min of 70 images
    
import Visualization

    
if __name__ == '__main__':
    Surface_deviation
    Analysis_parameters
    Plot_surfdev
    Plot_fitstack
    Plot_through
    Profile_extraction
    Plot_profiles
    Analysis_parameters_profile
    Plot_profile_parameters
    Through_length_equivalent_diameter
    Visualization
    
    