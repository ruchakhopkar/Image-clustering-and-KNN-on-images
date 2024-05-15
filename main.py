#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:44:14 2023

@author: ruchak
"""
import retrieve_catalog_data_better
import template_matching
import models
from tqdm import tqdm
save_path = '/extrastg/mdfs14/hamr-analysis/image_repo/rucha_test/'
waferlist = 	[ '6HHRU', '6HHRE', '6HHSK']	
folders_substring = ['LTE_EW', 'LSAL']
train = False
for wafer in tqdm(waferlist):
    print(wafer)
    #pulling images
    
    retrieve_catalog_data_better.main([wafer], folders_substring, folder_substring = True)
    
    #template matching
    df, test_feature_space = template_matching.main([wafer], folders_substring)
    
    #run models
    df = models.main(df, test_feature_space)
    
    df.to_csv(save_path + wafer + '.csv', index = False)

