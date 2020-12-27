import numpy as np
import imageio
from tensorflow_docs.vis import embed
import os
import cv2
import glob

model = "lsun"

path = r"./HoloGAN/samples/" + model

path2 = "./animations/animation_" + model + ".gif"
def loadImages(path):
    print('start')
    counter = 0
    arr = []
    for file in sorted(glob.glob(path + '/*.jpg')):
        print(file)
        img = imageio.imread(file)
        arr.append(img)
    arr_2 = np.flip(arr, 0)
    arr = np.append(arr, arr_2, axis=0)
    return arr

def animate(images):
    images = np.array(images)
    print(np.amax(images[0]))
    imageio.mimsave(path2, images)
    return embed.embed_file(path2)


images = loadImages(path)
animate(images)
