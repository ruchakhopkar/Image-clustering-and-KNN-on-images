#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:19:20 2023

@author: ruchak
"""

from bs4 import BeautifulSoup
import requests
import os
from io import BytesIO
from PIL import Image

link = 'https://sprpigtxl001.seagate.com/~gtx/wafer/proc_DSCVRIX/images/nfs/'
save_path = '/extrastg/mdfs14/hamr-analysis/image_repo/'

def get_folder_from_substring(wafer, folder):
    folders = []
    input_link = requests.get(link+wafer + '/', verify = False).text  
    soup = BeautifulSoup(input_link, 'lxml')
    results = soup.find_all('tr')

    for element in results:
        for subelements in element:
           x = subelements.find('a')
           if x is not None:
              if x.text.isupper():
               folders.append(x.text.split('/')[0])
    matching = [query for query in folders if any(sub in query for sub in folder)]
    return matching

def get_wafer(soup):
    wafers = []
    results = soup.find_all('tr')

    for element in results:
        for subelements in element:
           x = subelements.find('a')
           if x is not None:
               if (len(x.text)==6) & x.text.isupper():
                   current_wafer = x.text.split('/')[0]
                   wafers.append(current_wafer)
    return wafers

def get_folders(wafer, link):
    folders = []
    input_link = requests.get(link+wafer + '/', verify = False).text  
    soup = BeautifulSoup(input_link, 'lxml')
    results = soup.find_all('tr')

    for element in results:
        for subelements in element:
           x = subelements.find('a')
           if x is not None:
              if x.text.isupper():
               folders.append(x.text.split('/')[0])
    return folders

def get_images(save_path, link, wafer, f):
        input_link = requests.get(link+wafer + '/' + f + '/', verify = False).text
        soup = BeautifulSoup(input_link, 'lxml')
        results = soup.find_all('tr')

        for element in results:
            for subelements in element:
               x = subelements.find('a')

               if (x is not None):
                   if (x.text is not None) & (x.text[-2:] == '.1'):
                       url = link + wafer + '/' + f + '/' + x.text
                       try:
                           dest = url.replace('https://sprpigtxl001.seagate.com/~gtx/wafer/', save_path)
                           
                           if os.path.exists(os.path.join(dest.split('.')[0] + '.png')):
                               continue
                           im = Image.open(BytesIO(requests.get(url, verify = False).content))
                           
                           
                           os.makedirs(os.path.dirname(dest), mode = 0o777, exist_ok=True)
                           im.save(dest.split('.')[0] + '.png')
                           
           
            
                          
                       except Exception as e:
                           print(e)
                           
                           
def main(wafer_list, folder, folder_substring = False):
    
    input_link = requests.get(link, verify = False).text
    soup = BeautifulSoup(input_link, 'lxml')
    
    
    if not wafer_list:
        wafer_list = get_wafer(soup)
        
    for wafer in wafer_list:
        if folder_substring:
            folder = get_folder_from_substring(wafer, folder)
        if not folder:
            folder = get_folders(wafer, link)
        for f in folder:
            get_images(save_path, link, wafer, f)
                  
                   


