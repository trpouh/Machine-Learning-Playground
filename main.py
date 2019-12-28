from hawk import Hawk
from misc import Activations
from layers.convolution import ConvLayer, Kernels

import time

import cv2

if __name__ == "__main__":

    start = time.time()

    img = cv2.imread("IMG_0558.JPG", 3)
    img = cv2.resize(img, (500, 500))

    conv = ConvLayer(input_dimension=(500, 500, 3),
                     kernel=Kernels.EDGES, activation=Activations.ReLu)

    out = conv.fire(img)

    print("Duration: {:.3}s".format(time.time() - start))

    cv2.imshow('test', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hawk = Hawk()
    hawk.layer(conv)
