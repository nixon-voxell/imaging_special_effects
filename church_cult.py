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
butterfly = extend_image(butterfly, 100, 100, 0)
butterfly = resize_percentage(butterfly, 0.6)

LOOP_COUNT = 15
INV_LOOP_COUNT = 1.0 / LOOP_COUNT
butterfly_alpha = butterfly[:, :, 3]
butterfly_glow = cv2.Canny(butterfly_alpha, 50, 150)

for i in range(0, LOOP_COUNT):
    butterfly_glow = cv2.GaussianBlur(butterfly_glow, (5, 5), 10) + cv2.multiply(butterfly_glow, INV_LOOP_COUNT)

butterfly_glow = cv2.cvtColor(butterfly_glow, cv2.COLOR_GRAY2RGBA)
butterfly_glow[:, :, 3] = butterfly_glow[:, :, 1]
butterfly_glow = cv2.multiply(butterfly_glow, (1.0, 2.1, 3.4, 1))

for i in range(0, 3):
    overlay_transparent_to_transparent(
        butterfly,
        butterfly_glow,
    )

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
