import cv2
import numpy as np


def compute_depth(x1, x2, focal_length, baseline):
    """
    Compute the depth (z) using stereo vision.

    Parameters:
    x1 (float): x-coordinate of the point in the first image.
    x2 (float): x-coordinate of the corresponding point in the second image.
    focal_length (float): Focal length of the cameras.
    baseline (float): Separation distance between the two cameras (baseline).

    Returns:
    float: The computed depth (z) of the point.
    """
    disparity = x1 - x2
    if disparity == 0:
        raise ValueError("Disparity is zero, depth cannot be calculated.")

    z = (focal_length * baseline) / disparity
    return z


def select_points(image, window_name):
    """
    Select a point in the image manually.

    Parameters:
    image (ndarray): The image on which to select the point.
    window_name (str): Name of the window for point selection.

    Returns:
    tuple: x, y coordinates of the selected point.
    """
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) > 0:
        return points[0]
    else:
        raise ValueError("No point was selected.")


# Example usage:
# Load stereo images
left_image = cv2.imread('calibration_images\stereo_left\image_1.jpg')
right_image = cv2.imread('calibration_images\stereo_right\image_1.jpg')

# Select corresponding points in the images
x1, y1 = select_points(left_image, "Select point in left image")
x2, y2 = select_points(right_image, "Select point in right image")

# Camera parameters
focal_length = 1049  # in mm or pixels
baseline = 101  # in mm, meters, or the same unit as your focal length

# Calculate depth
z = compute_depth(x1, x2, focal_length, baseline)

print(f"Depth (z): {z}")
real_focal=(620*(x1-x2)/baseline)
print(real_focal)


# Optionally, display the images with selected points
cv2.imshow('Left Image with Point', left_image)
cv2.imshow('Right Image with Point', right_image)
cv2.waitKey(0)
cv2.destroyAllWindows()