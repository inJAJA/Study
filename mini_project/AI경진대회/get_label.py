import numpy as np
import re

def labeling(path):  
    f = open(path, 'r')
    txt = f.read()
    png_num = re.findall('jpg\s\d', txt) 
    label = []
    for num in png_num:
        num = int(num.replace('jpg ', ''))
        label.append(num)
    return(np.array(label))

path_train = '/tf/notebooks/Ja/data/train/train_label.txt'
path_val = '/tf/notebooks/Ja/data/validate/validate_label.txt'
# path_test = '/tf/notebooks/Ja/data/test/test_label.txt'

y_train= labeling(path_train)
y_val = labeling(path_val)

print(len(y_train)) # 240000
print(len(y_val))   # 80000

np.save('/tf/notebooks/Ja/data/y_train.npy', arr = y_train)
np.save('/tf/notebooks/Ja/data/y_test.npy', arr= y_val)
print('save_complete')


