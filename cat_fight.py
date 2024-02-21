import cv2
import numpy as np
from util.utils import *

background = read_file("roman-theatre-of-orange.jpg")
cats_fighting = read_file("cats-fighting.png")

background = resize_percentage(background, 0.5)
cats_fighting = resize_percentage(cats_fighting, 0.3)

overlay_transparent(
	background,
	cats_fighting
)

show_image(background)
