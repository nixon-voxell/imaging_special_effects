import cv2
import numpy as np
from util.utils import *

background = read_file("barcelona-church.jpg")
cat_standing = read_file("cat-standing.png")
butterfly = read_file("butterflies.png")
scratch_overlay = read_file("scratch-overlay2.png")

# Desaturate background
background = resize_percentage(background, 0.3)
background = desaturate(background, 0.3)
background[:, :, 0] = cv2.multiply(background[:, :, 0], 0.6)
background[:, :, 1] = cv2.multiply(background[:, :, 1], 0.7)

# Blur background
LOOP_COUNT = 3
for i in range(0, LOOP_COUNT):
    background = cv2.GaussianBlur(background, (3, 3), 2)

# Glow butterfly
butterfly = resize_percentage(butterfly, 0.6)
butterfly = glow_image(butterfly, 15, 100, (1, 2.1, 3.4, 1))

# Reduce scratch_overlay2 alpha
scratch_overlay[:, :, 3] = scratch_overlay[:, :, 3] * 0.4

print(f"background shape: {background.shape}")
print(f"cat_standing shape: {cat_standing.shape}")
print(f"butterfly shape: {butterfly.shape}")
print(f"scratch_overlay2 shape: {scratch_overlay.shape}")

overlay_transparent(
    background,
    cat_standing,
    int(background.shape[1] * 0.25) - 60,
    int(background.shape[0]) - 400,
)

overlay_transparent(
    background,
    butterfly,
    int(background.shape[1] * 0.25) - 30,
)

overlay_transparent(
    background,
    scratch_overlay,
)

show_image(background)
cv2.imwrite("church_cult.png", background)
