# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:22:07 2018

@author: Tian
"""
import os
import glob
import numpy as np
import json

data_path = 'E:\DateScienceRelated\Python_dl\Project/data'
testing_percentage = 20
image_info = 'E:\DateScienceRelated\Python_dl\Project/image_info.json'

def creat_image_lists(data_path, testing_percentage):
    '''
    将图片(无路径文件名)信息保存在字典中
    :param validation_percentage: 验证数据百分比
    :param testing_percentage:    测试数据百分比
    :return:                      字典{标签:{文件夹:str,训练:[],验证:[],测试:[]},...}
    '''
    result = {}
    sub_dirs = [x[0] for x in os.walk(data_path)]
    # 由于os.walk()列表第一个是'./'，所以排除
    is_root_dir = True  # <-----
    # 遍历各个label文件夹
    for sub_dir in sub_dirs:
        if is_root_dir:  # <-----
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # 遍历各个可能的文件尾缀
        for extension in extensions:
            # file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_glob = os.path.join(sub_dir, '*.' + extension)
            file_list.extend(glob.glob(file_glob))  # 匹配并收集路径&文件名
            # print(file_glob,'\n',glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()  # 生成label，实际就是小写文件夹名

        # 初始化各个路径&文件收集list
        training_images = []
        testing_images = []
#        validation_images = []

        # 去路径，只保留文件名
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据给验证和测试
            chance = np.random.randint(100)
            if chance < testing_percentage:
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 本标签字典项生成
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images
        }
    return result

if __name__ == '__main__':  
    image_list = creat_image_lists(data_path, testing_percentage)
    with open(image_info,'a') as f:
        json.dump(image_list,f)
        f.write('\n')
    f.close()    





    
    