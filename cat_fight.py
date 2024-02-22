import cv2
import numpy as np
from util.utils import *

background = read_file("roman-theatre-of-orange.jpg")
cats_fighting = read_file("cats-fighting.png")
scratch_overlay = read_file("scratch-overlay3.png")
crumpled_paper = read_file("crumpled-paper.jpg")
flare = read_file("flare.png")

background = resize_percentage(background, 0.5)
cats_fighting = resize_percentage(cats_fighting, 0.3)
flare = resize_percentage(flare, 0.6)

cats_fighting = glow_image(cats_fighting, 10, 100, (1, 2.6, 7.6, 1))
flare[:, :, 3] = flare[:, :, 3] * 0.7
flare[:, :, 1] = flare[:, :, 1] * 0.5
flare[:, :, 0] = flare[:, :, 0] * 0.6

crumpled_paper = crumpled_paper[
    :background.shape[0],
    :background.shape[1],
]

overlay_transparent(
    background,
    cats_fighting,
    int(background.shape[1] * 0.5) - 150,
)

overlay_transparent(
    background,
    flare,
    int(background.shape[1] * 0.25) - 100,
    int(background.shape[0] * 0.25) - 240,
)

overlay_transparent(
    background,
    scratch_overlay,
)

background = (background/255)*(crumpled_paper/255) * 255
background = background.astype(np.uint8)
# background += 0.05

show_image(background)
cv2.imwrite("cat_fight.png", background)
