from enum import Enum
import numpy as np

from skimage.exposure import rescale_intensity


class Activations(Enum):
    def ReLu(x): return x * (x > 0)

    def Normalize(x): return (rescale_intensity(
        x, in_range=(0, 255)) * 255).astype("uint8")
