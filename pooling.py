import numpy as np
import cv2
import img_tools

import time


img = cv2.imread("IMG_0558.JPG", 0)
img = img_tools.rescale(img, 100)


def pool(image, dim, func):

    (imH, imW) = image.shape

    new_img = np.zeros(((imH-1)//dim, (imW-1)//dim), dtype="float32")

    for y in np.arange(0, imH-dim, dim):
        for x in np.arange(0, imW-dim, dim):
            new_img[y//dim, x//dim] = func(image[y:y+dim, x:x+dim])

    return img_tools.normalize(new_img)


if __name__ == "__main__":

    start_time = time.time()

    img_min = pool(img, 16, lambda x: x[0][0])

    print("--- {:.3} seconds ({} ops per second) ---".format(time.time() -
                                                             start_time, int(1/(time.time() - start_time))))

    #img_max = pool(img, 8, lambda x: x.max())

    #img_avg = pool(img, 8, lambda x: np.average(x))

    cv2.imshow('min pool', img_min)
    #cv2.imshow('max pool', img_max)
    #cv2.imshow('avg pool', img_avg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
