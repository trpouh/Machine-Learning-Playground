import numpy as np

kernel = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
)

source = np.array(
    [
        [0, 0, 1, 0, 1],
        [2, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 2, 1, 0],
        [2, 1, 0, 1, 0]
    ]
)

(h, w) = source.shape
(kh, kw) = kernel.shape

conv = np.zeros((h-kh+1, w-kw+1))

for y in range(0, h-kh+1):
    for x in range(0, w-kw+1):
        roi = source[y:y+kh, x:x+kw]
        k = (roi * kernel).sum()
        conv[y, x] = k

print(conv)
