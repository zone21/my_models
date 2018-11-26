# -*- coding: utf-8 -*-
#  提取当前目录下所有文件及路径
import os
import random
from math import ceil, floor

num_class = 'resnet3'
t_f = open("./data/"+num_class+"/train_file_"+num_class+".txt", "a")
v_f = open("./data/"+num_class+"/val_file_"+num_class+".txt", "a")
te_f = open("./data/"+num_class+"/test_file_"+num_class+".txt", "a")
c_f = open("./data/"+num_class+"/class_"+num_class+".txt", "a")

file_train = '/data_ssd/resnet/crops2'
files = os.listdir(file_train)
num_img = 0
for subfile in files:
    os_listdir = os.listdir(os.path.join(file_train, subfile))

    if subfile != '0':
        num_img = len(os_listdir) if len(os_listdir) < 2000 else 2000
    else:
        num_img = len(os_listdir) if len(os_listdir) < 60000 else 60000

    if num_img < 200:
        print(subfile)
        # continue

    random.shuffle(os_listdir)
    print("subfile %s have %s" % (subfile, num_img))
    c_f.writelines(subfile + '\n')

    c_f.flush()
    t_f.flush()
    v_f.flush()
    te_f.flush()
    for file_index in range(0, int(ceil(num_img*1))):
        file = os_listdir[file_index]
        try:
            file = file.encode("gb18030").decode()
        except:
            try:
                file = file.encode("utf-8").decode()
            except:
                continue

        if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            t_f.writelines(os.path.join(file_train, subfile, file) + " " + subfile + '\n')

    # for file_index in range(int(floor(num_img * 0.9)),num_img):
    #     file = os_listdir[file_index]
    #     try:
    #         file = file.encode("gb18030").decode()
    #     except:
    #         try:
    #             file = file.encode("utf-8").decode()
    #         except:
    #             continue
    #
    #     if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
    #         v_f.writelines(os.path.join(file_train, subfile, file) + " " + subfile + '\n')

   # for file_index in range(floor(num_img * 0.8), num_img):
   #     file = os_listdir[file_index]
   #     try:
   #         file = file.encode("gb18030").decode()
   #     except:
   #         try:
   #             file = file.encode("utf-8").decode()
   #         except:
   #             continue

   #     if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
   #         te_f.writelines(os.path.join(file_train, subfile, file) + " " + subfile + '\n')

t_f.close()
v_f.close()
c_f.close()
te_f.close()
