import numpy as np
from matplotlib import pyplot as plt
import cv2


if __name__ == '__main__':
    # generate image frame sequence

    radius = 10
    angle = 5
    x0 = 23
    y0 = 34

    ims = []

    for i in range(10):
        im = np.zeros((100, 100), dtype=np.uint8)
        x1 = x0 + radius * np.cos(np.deg2rad(angle))
        y1 = y0 + radius * np.sin(np.deg2rad(angle))
        cv2.line(im, (x0, y0), (x1, y1), (255, 0, 0), 5)

        plt.imshow(im)
        plt.show()

        angle += 2

    # generate event stream from the images

