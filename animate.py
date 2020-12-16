import numpy as np
import imageio
from tensorflow_docs.vis import embed
import os
import cv2

model = "shapes3d"

path = r"./HoloGAN/samples/" + model

path2 = "./animations/animation_" + model + ".gif"
def loadImages(path):
    print('start')
    counter = 0
    arr = []
    for filename in os.listdir(path):
        img = cv2.imread(path+'/'+filename)
        arr.append(img)
        print('Image stored in array', counter)
        counter = counter + 1
    return arr

def animate(images):
    images = np.array(images)
    imageio.mimsave(path2, images)
    return embed.embed_file(path2)


images = loadImages(path)
animate(images)