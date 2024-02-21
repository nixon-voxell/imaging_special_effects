from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import cv2
import numpy as np

def apply_vintage_effect(image):
    # Convert image to grayscale
    grayscale_image = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(grayscale_image)
    grayscale_image = enhancer.enhance(1.5)

    # Convert image to RGB
    rgb_image = grayscale_image.convert("RGB")

    # Adjust color channels for sepia tone
    sepia_tone = (
        (242, 218, 179),  # Sepia tone values for RGB channels
        (204, 169, 116),
        (161, 102, 58)
    )

    # Apply sepia tone by mapping pixel values
    sepia_image = Image.new("RGB", rgb_image.size)
    for x in range(rgb_image.width):
        for y in range(rgb_image.height):
            pixel = rgb_image.getpixel((x, y))
            sepia_pixel = tuple(int(pixel[i] * c[i] / 255) for i in range(3) for c in sepia_tone)
            sepia_pixel = (sepia_pixel[0], sepia_pixel[1], sepia_pixel[2])  # Ensure it's a tuple of three elements
            sepia_image.putpixel((x, y), sepia_pixel)

    # Add vignette effect
    width, height = sepia_image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(30))
    sepia_image = Image.composite(sepia_image, Image.new("RGB", (width, height), (255, 255, 255)), mask)

    return sepia_image

# Load your historical building image
image = Image.open("historical_building.jpg")

# Apply vintage effect
# vintage_image = apply_vintage_effect(image)
# vintage_image.show()

def apply_oil_painting_effect(image):
    # Apply oil painting filter
    oil_painting = image
    oil_painting = image.filter(ImageFilter.EDGE_ENHANCE)

    # # Adjust brightness and contrast
    # enhancer = ImageEnhance.Brightness(oil_painting)
    # oil_painting = enhancer.enhance(1.2)
    # enhancer = ImageEnhance.Contrast(oil_painting)
    # oil_painting = enhancer.enhance(1.5)

    # Add canvas texture
    canvas_texture = Image.open("canvas_texture.jpg")
    canvas_texture = canvas_texture.resize(image.size)  # Resize to match the size of the original image
    # oil_painting.paste(canvas_texture, (0, 0))

    return oil_painting

# Load your historical building image
image = Image.open("historical_building.jpg")

# Apply oil painting effect
# oil_painting_image = apply_oil_painting_effect(image)
# oil_painting_image.show()

def apply_comic_effect(image):
    # Convert image to LAB color space
    edge_image = image.filter(ImageFilter.EMBOSS)
    lab_image = image.convert("LAB")

    # Separate LAB channels
    L, A, B = lab_image.split()

    # Desaturate A and B channels
    L = L.point(lambda x: x * 0.7)
    A = A.point(lambda x: x * 0.5)
    B = B.point(lambda x: x * 0.8)

    # Merge LAB channels back
    lab_image = Image.merge("LAB", (L, A, B))

    # Convert LAB image back to RGB
    comic_image = lab_image.convert("RGB")

    return edge_image

# Load your image
image = Image.open("historical_building.jpg")

# Apply comic effect
comic_image = apply_comic_effect(image)
comic_image.show()
