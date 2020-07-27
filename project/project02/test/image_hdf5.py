import numpy as np
import h5py
from sklearn.model_selection import train_test_split

f = h5py.File('./project/project02/data/image_hdf5.hdf5','w')

bi = np.load('./project/project02/data/final/face_image_Bichon_frise.npy')
bc = np.load('./project/project02/data/final/face_image_Border_collie.npy')
bl = np.load('./project/project02/data/final/face_image_Bulldog.npy')
ci = np.load('./project/project02/data/final/face_image_Chihuahua.npy')
cg = np.load('./project/project02/data/final/face_image_Corgi.npy')
ds = np.load('./project/project02/data/final/face_image_Dachshund.npy')
gr = np.load('./project/project02/data/final/face_image_Golden_retriever.npy')
hs = np.load('./project/project02/data/final/face_image_Husky.npy')
jd = np.load('./project/project02/data/final/face_image_Jindo_dog.npy')
mt = np.load('./project/project02/data/final/face_image_Maltese.npy')
pg = np.load('./project/project02/data/final/face_image_Pug.npy')
yt = np.load('./project/project02/data/final/face_image_Yorkshire_terrier.npy')

c = f.create_group('category')

bi = c.create_dataset('Bichon', data = bi)
bc = c.create_dataset('Border', data = bc)
bl = c.create_dataset('Bull', data = bl)
ci = c.create_dataset('Chihuahua', data = ci)
cg = c.create_dataset('Corgi', data = cg)
ds = c.create_dataset('Dachshund', data = ds)
gr = c.create_dataset('Retriever', data = gr)
hs = c.create_dataset('Husky', data = hs)
jd = c.create_dataset('Jindo', data = jd)
mt = c.create_dataset('Maltese', data = mt)
pg = c.create_dataset('Pug', data = pg)
yt = c.create_dataset('Yorkshire', data = yt)

# 
image = f.create_dataset('image', (5530,128, 128, 3), dtype='float64')

start = 0

for i, x in enumerate([bi, bc, bl, ci, cg, ds, gr, hs, jd, mt, pg, yt]):
    end = start + x.shape[0]
    print(end)

    if i == 0:
        image[:end] = x
        start = x.shape[0]

    else:
        image[start:end] = x
        start = end

print(image.shape)

f.close()

print('----------------------')

r = h5py.File('./project/project02/data/image_hdf5.hdf5','r')
print(r)
print(r.keys())

# group = r.get('image')
# print(group.items())
# print(group)
# print(type(group))

from keras.utils.io_utils import HDF5Matrix
