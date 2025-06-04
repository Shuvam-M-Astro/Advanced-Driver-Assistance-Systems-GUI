import cv2
import matplotlib.pyplot as plt
from config import DISPLAY_DIR
from models.sign_classifier import classify_traffic_sign
from detectors.traffic_light import detect_traffic_light
from detectors.lane_detector import detect_lane
from utils.gui_overlay import overlay_gui
import os

for i in range(525, 526):
    path = f"Test/{i}.png"
    img = cv2.imread(path)

    tl_state = detect_traffic_light(img)
    sign_idx = classify_traffic_sign(img)
    lane_img = detect_lane(cv2.imread(path))
    sign_display_path = os.path.join(DISPLAY_DIR, f"{sign_idx}.png")

    if tl_state == 'green':
        gui_path = 'GUI_green.jpg'
    elif tl_state == 'red':
        gui_path = 'GUI_red.jpg'
    else:
        gui_path = 'GUI_yellow.jpg'

    gui_img = overlay_gui(sign_display_path, gui_path)

    # Save outputs
    cv2.imwrite('Lane.png', lane_img)
    cv2.imwrite('GUI_res.png', gui_img)

    # Display
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)); axs[0, 0].set_title("Input"); axs[0, 0].axis('off')
    axs[0, 1].imshow(cv2.imread('Lane.png'), cmap='gray'); axs[0, 1].set_title("Lane Detection"); axs[0, 1].axis('off')
    axs[1, 0].imshow(cv2.cvtColor(cv2.imread('GUI_res.png'), cv2.COLOR_BGR2RGB)); axs[1, 0].set_title("GUI"); axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    plt.savefig(f"output/{i}_final.jpg")