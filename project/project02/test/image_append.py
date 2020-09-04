import h5py
import numpy

# t = h5py.File('D:\Jain\\20200805/image_hdf5.hdf5', 'r')
# t = t['512'][:]
# print(t.shape)
t = np.load('D:/data/face_image_total.npy')
print(t.shape)

#------------------------------------------------------
d = h5py.File('D:/data/face_Doberman.hdf5', 'r')
d = d['doberman'][:431]

i1 = h5py.File('D:/data/face_Italian_Greyhound.hdf5', 'r')
i1 = i1['Italian_Greyhound'][:390]

i2 = h5py.File('D:/data/face_Italian_Greyhound_part2.hdf5', 'r')
i2 = i2['Italian_Greyhound'][:109]

p = h5py.File('D:/data/face_Pekingese.hdf5', 'r')
p = p['pekingese'][:463]

s = h5py.File('D:/data/face_Sichu.hdf5', 'r')
s = s['Sichu'][:558]
#------------------------------------------------------



with h5py.File('D:/data/DATA.hdf5','a') as f:
    imageset = f.create_dataset('512', (7481, 512, 512, 3), maxshape = (None, 512, 512, 3))

    start = 0

    for i, x in zip([5530, 431, 390, 109, 463, 558],[t, d, i1, i2, p, s]):
        end = start + i
        print(end)

        imageset[start:end] = x

        start = end

    print('------------ complete -----------')

# t = h5py.File('D:/data/image_hdf5.hdf5', 'r')
# t = t['label'][:]
t = np.load('D:/data/face_label_total.npy')
print(t.shape)

with h5py.File('D:/data/DATA.hdf5','a') as f:
    # del f['label']
    label = f.create_dataset('label', (7481, ), maxshape = (None, ))
    start = 0
    k = 12

    for i in [5530, 431, 390+109, 463, 558]:
        end = start + i
        print(end)

        if i == 5530:
            label[start:end] = t
        else:
            label[start:end] = k
            k += 1

        start = end

    print('------------ complete -----------')