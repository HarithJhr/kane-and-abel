import cv2
import numpy as np

def detect_high_obstacles(state: np.ndarray, unittest_mode: bool = False, unittest_return_val: str = "null") -> str:
    # This is for unit testing purposes
    if unittest_mode:
        return unittest_return_val

    if type(state) is not np.ndarray:
        return "NONE"
    
    # Convert state to grayscale
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Define the detection region
    top = 160
    bottom = 210
    left = 130
    right = 190
    detection_region = gray_state[top:bottom, left:right]

    # Threshold the detection region
    _, thresholded_detection_region = cv2.threshold(detection_region, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded detection region
    contours, _ = cv2.findContours(thresholded_detection_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Contour detection
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        if 1 < w < 10 and 4 < h < 15:
            return "ENEMY"
        
        if w > 1 and h > 15:
            return "PIPE"
        
    return "NONE"


def detect_valley(state: np.ndarray, unittest_mode: bool = False, unittest_return_val: bool = False) -> bool:
    # This is for unit testing purposes
    if unittest_mode:
        return unittest_return_val

    if type(state) is not np.ndarray:
        return False
    
    # Convert state to grayscale
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Define the detection region
    top = 210
    bottom = 240
    left = 140
    right = 145
    detection_region = gray_state[top:bottom, left:right]

    # Threshold the detection region
    _, thresholded_detection_region = cv2.threshold(detection_region, 10, 255, cv2.THRESH_BINARY)

    # Find gaps in the floor
    valley_detection = np.sum(thresholded_detection_region == 0, axis=0) > 50


    return np.all(thresholded_detection_region==0)