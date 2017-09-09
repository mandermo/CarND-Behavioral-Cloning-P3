import cv2
import numpy as np

# Takes BGR image with 8 bit per channel (0 to 255) and convert it to YUV that
# is normalized to -1. to 1 float32
def preprocess_img(img):
    # Convert to float and normalize to 0 to 1
    # Just doing divide converts to float64 which BGR2YUV dislikes.
    img = img.astype(np.float32)
    img = img / 255.
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    # According to OpenCV doc, values are between 0 to 1

    # Convert to -1 .. 1 to make it training better.
    img = 2. * img - 1.
    return img
