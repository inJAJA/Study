import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
from tkinter import Tcl
import os

filename = os.listdir("D:/Study/project/GAN/result/conv2d")
f = Tcl().call('lsort', '-dict', filename)                  # filename sort
print(f)

path = [f"D:/Study/project/GAN/result/conv2d/{i}" for i in f]
paths = [ Image.open(i) for i in path]
imageio.mimsave('D:/Study/project/GAN/result/pngs/test.gif', paths, fps=0.9)