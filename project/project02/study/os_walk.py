import os
import re
import numpy as np

path = 'C:/Users/bitcamp/Downloads/images'

# os.walk
for root, dir, files in os.walk(path):

    print('1 =', root)      # folder path
    print('2 ==',dir)       # ?
    print('3 ===',files)    # folder내 files

# os.listdir
folder_name = os.listdir(path)
print(folder_name)

