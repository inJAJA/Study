import h5py
import cv2

with h5py.File('C:/Users/bitcamp\Desktop/breed_add/hdf5/pekingese.hdf5', 'r') as f:
    img = f['pekingese'][100]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', img)
    cv2.waitKey(0)