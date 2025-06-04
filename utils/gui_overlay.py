import cv2
import cvzone

def overlay_gui(sign_img_path, gui_img_path, sign_scale=15, gui_scale=60, position=(750, 370)):
    """
    Overlays a resized sign image onto a GUI image.

    Parameters:
        sign_img_path (str): Path to the sign image (with alpha channel).
        gui_img_path (str): Path to the GUI background image.
        sign_scale (int): Scale percentage for the sign image.
        gui_scale (int): Scale percentage for the GUI background.
        position (tuple): (x, y) coordinates to place the sign image on the GUI.

    Returns:
        result_img (ndarray): The image with overlay.
    """
    sign_img = cv2.imread(sign_img_path, cv2.IMREAD_UNCHANGED)
    gui_img = cv2.imread(gui_img_path, cv2.IMREAD_UNCHANGED)

    if sign_img is None:
        raise FileNotFoundError(f"Sign image not found at path: {sign_img_path}")
    if gui_img is None:
        raise FileNotFoundError(f"GUI image not found at path: {gui_img_path}")

    # Resize images
    sign_dim = (int(sign_img.shape[1] * sign_scale / 100), int(sign_img.shape[0] * sign_scale / 100))
    gui_dim = (int(gui_img.shape[1] * gui_scale / 100), int(gui_img.shape[0] * gui_scale / 100))

    sign_resized = cv2.resize(sign_img, sign_dim, interpolation=cv2.INTER_AREA)
    gui_resized = cv2.resize(gui_img, gui_dim, interpolation=cv2.INTER_AREA)

    # Overlay
    result_img = cvzone.overlayPNG(gui_resized, sign_resized, position)
    return result_img
