from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import time
import img_tools

edges = np.array(([-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]), dtype="int")

vlines = np.array(([0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]), dtype="int")


blur = (1/16) * np.array(([1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]), dtype="int")


axis_l = np.array(([-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]), dtype="int")

axis_r = np.array(([1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]), dtype="int")

axis_vert = np.array(([1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]), dtype="int")

relief = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype="int")


def convolve(img, kernel, stride, padding, act=lambda i: (rescale_intensity(i, in_range=(0, 255)) * 255).astype("uint8")):

    (imH, imW) = img.shape
    (kH, kW) = kernel.shape

    (padH, padW) = ((kH-1) // 2, (kW-1) // 2)

    img = cv2.copyMakeBorder(img, padH, padW, padH, padW, cv2.BORDER_REPLICATE)

    new_img = np.zeros(img.shape, dtype="float32")

    for y in np.arange(padH, imH + padH):
        for x in np.arange(padW, imW + padW):

            roi = img[y - padH:y + padH + 1, x - padW:x + padW + 1]
            k = (roi * kernel).sum()
            new_img[y - padH, x - padW] = k

    return act(conv)


if __name__ == "__main__":

    img = cv2.imread("IMG_0558.JPG", 3)
    resized = img_tools.rescale(img, 10)

    cv2.imshow('test', resized)

    cv2.imshow('test a', resized[:, :, 0])
    cv2.imshow('test b', resized[:, :, 1])
    cv2.imshow('test c', resized[:, :, 2])

    # im = convolve(resized, relief)
    # im = convolve(resized, relief, lambda x: np.maximum(0, x))
    # print(im)

    # cv2.imshow('img original', resized)

    #cv2.imshow('img relief', convolve(resized, edges))
    # cv2.imshow('img relief max', convolve(
    #    resized, axis_r, lambda x: np.maximum(0, x)))
    # cv2.imshow('img edges', convolve(resized, edges))
    # cv2.imshow('img blur', convolve(resized, blur))
    # cv2.imshow('img axis right', convolve(resized, axis_r))
    # cv2.imshow('img axis left', convolve(resized, axis_l))
    # cv2.imshow('img axis vert', convolve(resized, axis_vert))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
