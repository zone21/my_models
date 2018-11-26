import os

with open('./134_class.txt') as cf:
    class_list = [c.strip() for c in cf.readlines()]
cf.close()

f = open("./other_class.txt", "w")
file_test = '/data_ssd/product_20181012124806/testSet/'
files = os.listdir(file_test)
for subfile in files:
    if subfile in class_list:
        continue

    f.writelines(subfile + '\n')
    # num_img = len(os.listdir(os.path.join(file_test, subfile)))
    # if num_img > 200:
    #     f.writelines(subfile+'\n')
    #     print(subfile)

f.close()
