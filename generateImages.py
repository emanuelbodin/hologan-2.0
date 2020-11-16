from shape3D import *
from PIL import Image
import os

data = load_dataset()
counter = 3001
for i in range(500):
    batch = sample_random_batch(100, data)
    for img in batch:
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        new_image = Image.new('RGB',(img.size[1], img.size[1]), (250,250,250))
        img.paste(img)
        img.save(os.path.join("../datasets/shapes3d", "{0}.jpg".format(counter)),"JPEG")
        counter = counter + 1
    print("Image saved: ", counter)    