# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 05:27:39 2018

@author: Tian
"""
import os
import shutil
import json
from PIL import Image

IMAGE_INFO = './image_info.json'
DATA_PATH = './data'

def get_image_info(image_info):
    
    info = list()
    with open(image_info,'r') as f:
        for line in f:
            info.append(json.loads(line))
    f.close()
    image_list = info[0]
    
    return image_list


def dataResize(image_list,DATA_PATH):
	for label in image_list.keys():
		print(type(label))
	#    os.mkdir(DATA_PATH + '/test' + '/' + label)
	#    os.mkdir(DATA_PATH + '/train' + '/' + label)
		for file in image_list[label]['testing']:
			newpath = DATA_PATH + '1/test' + '/' + label + '/' + file
			if not os.path.exists(newpath):
				os.mkdir(newpath)			
			oldpath = DATA_PATH + '/' + label + '/' + file
			img = Image.open(oldpath)
			img = img.resize(28,28)
			img = img.save(newpath)
		for file in image_list[label]['training']:
			newpath = DATA_PATH + '1/train' + '/' + label + '/' + file
			if not os.path.exists(newpath):
				os.mkdir(newpath)
			oldpath = DATA_PATH + '/' + label + '/' + file
			img = Image.open(oldpath)
			img = img.resize(28,28)
			img = img.save(newpath)

if __name__ == '__main__':
	image_list = get_image_info(IMAGE_INFO)    
	dataResize(image_list,DATA_PATH)

        

          