import math
import matplotlib.pyplot as plt
from Customdataset import Customdataset

face_dataset = Customdataset()
def visualize(samples = 32):
    grid_size = math.sqrt(samples)
    grid_size = math.ceil(grid_size)
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    for i in range(samples):
        ax = fig.add_subplot(grid_size, grid_size, i+1, xticks=[], yticks=[])
        image, landmark = face_dataset[i]
        image = image.squeeze().permute(1,2,0)
        plt.imshow(image)
        kpt = []
        for j in range(68):
            kpt.append(plt.plot(landmark[j][0], landmark[j][1], 'g.'))
    plt.tight_layout()
    plt.show()

visualize()