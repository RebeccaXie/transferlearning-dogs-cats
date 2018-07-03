# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 09:41:23 2018

@author: Tian
"""
import os
import shutil

def createDir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("创建文件夹失败")
            exit(1)

path="E:/DateScienceRelated/Python_dl/Project/rawdata/data"
    
dogs=[path+'/'+i for i in os.listdir(path) if 'dog' in i]

cats=[path+'/'+i for i in os.listdir(path) if 'cat' in i]

createDir(path+"/train_1/dogs")
createDir(path+"/train_1/cats")
createDir(path+"/test_1/dogs")
createDir(path+"/test_1/cats")


for dog,cat in list(zip(dogs,cats))[:100]:
    shutil.copyfile(dog,path+"/train_1/dogs/"+os.path.basename(dog))
    print(os.path.basename(dog)+"操作成功")
    shutil.copyfile(cat, path + "/train_1/cats/" + os.path.basename(cat))
    print(os.path.basename(cat) + "操作成功")
for dog, cat in list(zip(dogs, cats))[100:120]:
    shutil.copyfile(dog, path + "/test_1/dogs/" + os.path.basename(dog))
    print(os.path.basename(dog) + "操作成功")
    shutil.copyfile(cat, path + "/test_1/cats/" + os.path.basename(cat))
    print(os.path.basename(cat) + "操作成功")



