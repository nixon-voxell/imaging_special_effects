import cv2
import numpy as np
from util.utils import *

background = read_file("barcelona-church.jpg")
background = resize_percentage(background, 0.3)
background = desaturate(background, 0.3)
background[:, :, 0] = cv2.multiply(background[:, :, 0], 0.6)
background[:, :, 1] = cv2.multiply(background[:, :, 1], 0.7)

cat_standing = read_file("cat-standing.png")

butterfly = read_file("butterfly.png")
butterfly = extend_image(butterfly, 100, 100, 0)
butterfly = resize_percentage(butterfly, 0.2)

# glow butterfly
LOOP_COUNT = 20
INV_LOOP_COUNT = 1.0 / LOOP_COUNT
butterfly_alpha = butterfly[:, :, 3]
butterfly_glow = cv2.Canny(butterfly_alpha, 50, 150)

for i in range(0, LOOP_COUNT):
    butterfly_glow = cv2.GaussianBlur(butterfly_glow, (5, 5), 10) + cv2.multiply(butterfly_glow, INV_LOOP_COUNT)

butterfly_glow = cv2.cvtColor(butterfly_glow, cv2.COLOR_GRAY2RGBA)
butterfly_glow[:, :, 3] = butterfly_glow[:, :, 1]
butterfly_glow = cv2.multiply(butterfly_glow, (255, 255, 0, 1))

for i in range(0, 3):
    overlay_transparent_to_transparent(
        butterfly,
        butterfly_glow,
    )

print(f"background shape: {background.shape}")
print(f"cat_standing shape: {cat_standing.shape}")
print(f"butterfly shape: {butterfly.shape}")

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
    # int(background.shape[0])
)

# cv2.imshow("image", butterfly)
cv2.imshow("image", background)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()
