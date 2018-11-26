import os
num_class = '624'
c_f = open("../data/"+num_class+"/class_"+num_class+".txt", "r", encoding='GB18030')
z_f = open("../data/0_"+num_class+".txt", "w", encoding='GB18030')

class_list = [c.strip() for c in c_f.readlines()]

num = 10
file_test = '/data/data/product/testSet/'
files = os.listdir(file_test)
for subfile in files:
    if subfile in class_list:
        continue

    # num_img = len(os.listdir(os.path.join(file_test, subfile)))
    try:
        imgfiles = os.listdir(os.path.join(file_test, subfile))
    except:
        continue

    count = 0
    for imgfile in imgfiles:
        if count >= num:
            break

        if imgfile.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            z_f.writelines(os.path.join(subfile, imgfile) + '\n')
            count += 1

z_f.close()
c_f.close()
