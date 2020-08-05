import h5py
import cv2

# with h5py.File('C:/Users/bitcamp\Desktop/breed_add/hdf5/face_Doberman.hdf5', 'a') as f:
with h5py.File('D:/data/face_Italian_Greyhound_part2.hdf5', 'a') as f:

    print('key값 :',f.keys())

    img = f['Italian_Greyhound'][50]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', img)
    cv2.waitKey(0)
