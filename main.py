from hawk import Hawk
from misc import Activations
from layers.convolution import ConvLayer, Kernels

import time

import cv2

if __name__ == "__main__":

    (w, h) = (250, 250)

    img = cv2.imread("IMG_0558.JPG", 1)
    #img = cv2.imread("bbenson.jpg", 1)
    img = cv2.resize(img, (w, h))

    conv = ConvLayer(input_dimension=(w, h, 1),
                     kernel=Kernels.SHARP, activation=Activations.ReLu)

    start = time.time()

    out = conv.fire(img)

    print("Duration: {:.3}s".format(time.time() - start))

    cv2.imshow('test', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hawk = Hawk()
    hawk.layer(conv)
