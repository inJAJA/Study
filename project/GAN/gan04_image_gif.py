import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
from tkinter import Tcl
import os

file_dir = "D:/Study/project/GAN/result/lsgan"

filename = os.listdir(file_dir)
f = Tcl().call('lsort', '-dict', filename)                  # filename sort
# print(f)

path = [file_dir+f"/{i}" for i in f]
paths = [ Image.open(i) for i in path]

# save
save_name = 'lsgan_02'
imageio.mimsave('D:/Study/project/GAN/result/pngs/%s.gif'%(save_name), paths, fps=2)