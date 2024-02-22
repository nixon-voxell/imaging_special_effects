import cv2
import numpy as np

def read_file(filename: str) -> cv2.typing.MatLike:
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    return img

def resize_percentage(image: cv2.typing.MatLike, percentage: float) -> cv2.typing.MatLike:
    return cv2.resize(image, (0, 0), fx = percentage, fy = percentage)

def extend_image(image: np.ndarray, height: int, width: int, color):
    half_height = int(height / 2)
    half_width = int(width / 2)

    image_extended = np.ndarray((image.shape[0] + height,) + (image.shape[1] + width,) + image.shape[2:], dtype=image.dtype)
    image_extended[:, :] = color
    image_extended[
        half_height:image.shape[0] + half_height,
        half_width:image.shape[1] + half_width, :
    ] = image

    return image_extended

def add_alpha_channel(image: cv2.typing.MatLike, alpha_data = 0.0):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    img[:, :, 3] = alpha_data

    return img

def desaturate(image: cv2.typing.MatLike, value = 0.5):
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * value

    # Convert back to BGR
    img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return img

def overlay_transparent(background, foreground, x_offset=None, y_offset=None):
    """
    Overlay a forground onto the background (background will be modified)
    """

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def overlay_transparent_to_transparent(background, foreground, x_offset=None, y_offset=None):
    """
    Overlay a foreground onto the background (background will be modified)
    """

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 4, f'background image should have exactly 4 channels (RGBA). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    # calculate overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, -x_offset)
    fg_y = max(0, -y_offset)
    w = min(fg_w - fg_x, bg_w - bg_x)
    h = min(fg_h - fg_y, bg_h - bg_y)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha channels from foreground and background images
    background_alpha = background_subsection[:, :, 3] / 255  # 0-255 => 0.0-1.0
    foreground_alpha = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # combine the alphas to get the final alpha channel for the composite image
    composite_alpha = foreground_alpha + background_alpha * (1 - foreground_alpha)

    # calculate the composite color channels
    composite_colors = (
        foreground[:, :, :3] * foreground_alpha[..., None] + 
        background_subsection[:, :, :3] * background_alpha[..., None] * (1 - foreground_alpha[..., None])
    )

    # combine the alpha and color channels into the composite image
    composite = np.empty((h, w, 4), dtype=np.uint8)
    composite[:, :, :3] = composite_colors
    composite[:, :, 3] = composite_alpha * 255

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def glow_image(image, loop: int, extend: int, color_multiplier):
    image = extend_image(image, extend, extend, 0)

    inv_loop = 1.0 / loop
    image_alpha = image[:, :, 3]
    image_glow = cv2.Canny(image_alpha, 50, 150)

    for i in range(0, loop):
        image_glow = cv2.GaussianBlur(image_glow, (5, 5), 10) + cv2.multiply(image_glow, inv_loop)

    image_glow = cv2.cvtColor(image_glow, cv2.COLOR_GRAY2RGBA)
    image_glow[:, :, 3] = image_glow[:, :, 1]
    image_glow = cv2.multiply(image_glow, color_multiplier)

    for i in range(0, 3):
        overlay_transparent_to_transparent(
            image,
            image_glow,
        )

    return image

def show_image(image):
    # cv2.imshow("image", butterfly)
    cv2.imshow("image", image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()
