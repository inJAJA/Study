import numpy as np
import h5py

a=np.random.random(size=(100,20))

with h5py.File('./project/project02/data_hdf5.hdf5', 'w') as f: # write
    f.create_dataset('dataset_1',  data=a)                      # numpy 형식 data저장

with h5py.File('./project/project02/data_hdf5.hdf5', 'r') as f: # read
    data = f['dataset_1'][:]                                    # 'dataset_1'불러오기

print(data)


# 하나하나 저장
f = h5py.File('./project/project02/data_hdf5.hdf5', 'w')    # hdf5 file생성
f.create_dataset('image', (100, 20), dtype='float64')       # 'image'라는 (100, 20)의 빈 공간 생성 
dataset = f['image']                                        # 'image'저장소 불러오기

for i in range(len(a)):
    data = a[i]
    dataset[i] = data

print(dataset[10].shape)                                    # (20,)