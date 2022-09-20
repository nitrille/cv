# Import dependencies
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Image path
img_path = os.path.join(os.path.dirname(__file__), 'test.jpg')

# Load image
img = cv2.imread(img_path)

# Split the image into the three colour channels
red, green, blue = cv2.split(img)

# Compose the image in the RGB colour space
img1 = cv2.merge([red, green, blue])

# Compose the image in the RBG colour space
img2 = cv2.merge([red, blue, green])

# Compose the image in the GRB colour space
img3 = cv2.merge([green, red, blue])

# Compose the image in the BGR colour space
img4 = cv2.merge([blue, green, red])

# Create the collage
out1 = np.hstack([img1, img2])
out2 = np.hstack([img3, img4])
out = np.vstack([out1, out2])

# Plot the collage
cv2.imshow('test', out)

# Save image to file
# result_path = os.path.join(os.path.dirname(__file__), 'result.jpg')
# cv2.imwrite(result_path, out)

# Plot images
# cv2.imshow('RGB', img1)
# cv2.imshow('RBG', img2)
# cv2.imshow('GRB', img3)
# cv2.imshow('BGR', img4)

# Waiting any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()