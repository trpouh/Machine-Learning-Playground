from enum import Enum
import numpy as np
import cv2


class ConvLayer:
    config = {}

    def __init__(self, input_dimension, kernel, activation, stride=1, padding=True):
        self.config = {
            'in_dim': input_dimension,
            'kernel': kernel.value.arr,
            'act': activation,
            'stride': stride,
            'padding': padding
        }

    def out_shape(self):

        (kh, kw) = self.config['kernel'].shape

        (pH, pW) = self.pad()

        (y, x, z) = self.config['in_dim']

        s = self.config['stride']

        return (int(((y - kh + 2 * pH) / s) + 1), int(((x - kw + 2 * pW) / s) + 1), z)

    def pad(self):
        (kh, kw) = self.config['kernel'].shape
        return ((kh - 1) // 2, (kw - 1) // 2) if self.config['padding'] else (0, 0)

    def fire(self, input_data):

        (h, w, d) = self.config['in_dim']

        kernel = self.config['kernel']

        (kH, kW) = kernel.shape

        (padH, padW) = self.pad()

        img = cv2.copyMakeBorder(
            input_data, padH, padW, padH, padW, cv2.BORDER_REPLICATE)

        new_img = np.zeros((h, w, d), dtype="float32")

        for z in np.arange(0, d):

            im_dim = img[:, :, z]

            for y in np.arange(0, h+padH - h % kW, self.config['stride']):
                for x in np.arange(0, w+padW - w % kH, self.config['stride']):

                    #print("Kernel: {} x {}".format(kH, kW))

                    roi = im_dim[y:y+kH, x:x+kW]

                    # print("pos(y={};x={}); Roi: {} ({})".format(
                    #    y, x, str(roi), roi.shape))

                    #roi = im_dim[y - padH:y + padH + 1, x - padW:x + padW + 1]
                    k = (roi * kernel).sum()
                    new_img[y - padH, x - padW, z] = k

        return self.config['act'](new_img)


class Kernel(object):

    arr = np.zeros(0)

    def __init__(self, arr):
        self.arr = arr

    def __repr__(self):
        return "Kernel({})".format(np.array_repr(self.arr))


class Kernels(Enum):
    EDGES = Kernel(
        np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype="int"))

    RELIEF = Kernel(
        np.array(([-2, -1, 0], [-1, 1, 1], [0, 1, 2]), dtype="int"))
