import os
import re

dir_name = 'images/LR/'
img_name = os.listdir(dir_name)
# x = dir_name+img_name[0]
for img in img_name:
    os.renames(dir_name+img, dir_name+re.sub('x4d', '', img, count=0))
