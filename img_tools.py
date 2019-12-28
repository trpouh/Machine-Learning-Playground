from skimage.exposure import rescale_intensity
import cv2


def rescale(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def normalize(img):
    return (rescale_intensity(img, in_range=(0, 255)) * 255).astype("uint8")
