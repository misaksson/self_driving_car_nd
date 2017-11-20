"""Helper tool to find perspective transform for birds-eye-view.

Let the user select perspective source points in an image with lanes lines,
and then measure lines dash length and lane width, which provides for
calculation of perspective transformation matrix. The calculation assumes
lane properties as in the USA, see constants section below.

The output from this tool is a pickle file with the transformation matrix
together with the x and y pixel resolution in meters.
"""
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
import sys
sys.path.append("../src/")

# Constants
dash_length_in_meters = 3.048   # 10 feet
lane_width_in_meters = 3.6576  # 12 feet
expected_visual_length_in_meters = 30.0  # Defines the transformed view length.
expected_visual_width_in_meters = 10.0  # Defines the transformed view width.


# mouse callback function
def get_coord(event, x, y, flags, param):
    global coords
    global done
    global output_image
    global temp_image
    if event == cv2.EVENT_LBUTTONUP:
        coords.append((x, y))
        if len(coords) > 1:
            cv2.line(output_image, coords[-2], coords[-1], color=(0, 0, 255), thickness=1)

    temp_image = np.copy(output_image)
    if event == cv2.EVENT_MOUSEMOVE:
        if len(coords) > 0:
            cv2.line(temp_image, coords[-1], (x, y), color=(0, 0, 255), thickness=1)


def load_camera_calibration():
    with open("../src/calibration.p", 'rb') as fid:
        return pickle.load(fid)


calibration = load_camera_calibration()

# Get a frame 17 seconds into the sequence, where the road is fairly straight.
input_image = VideoFileClip('../input/project_video.mp4').get_frame(17)
input_image = calibration.undistort(input_image)
input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
h, w, _ = input_image.shape

# Show image and let user select four coordinates for perspective transform.
draw_win = 'Select source points for perspective transform'
cv2.namedWindow(draw_win)
cv2.setMouseCallback(draw_win, get_coord)
output_image = np.copy(input_image)
temp_image = np.copy(output_image)
coords = []
while 1:
    cv2.imshow(draw_win, temp_image)
    cv2.waitKey(20)
    if len(coords) >= 4:
        break

cv2.line(output_image, coords[-1], coords[0], color=(0, 0, 255), thickness=1)
cv2.imshow(draw_win, output_image)
cv2.imwrite('../output/perspective_source.png', output_image)

src = np.array([list(coord) for coord in coords]).astype(np.float32)
dst_width = 500
dst_height = h
dst = np.float32([[(w - dst_width) // 2, h],
                  [(w - dst_width) // 2, h - dst_height],
                  [(w + dst_width) // 2, h - dst_height],
                  [(w + dst_width) // 2, h]])

# Temporary transform used for measurements.
transformation_matrix = cv2.getPerspectiveTransform(src, dst)
transformed_image = cv2.warpPerspective(input_image, transformation_matrix, (w, h), flags=cv2.INTER_LINEAR)

coords = []
output_image = np.copy(transformed_image)
temp_image = np.copy(transformed_image)
measure_win = 'Measure dash length and lane width'
cv2.namedWindow(measure_win)
cv2.setMouseCallback(measure_win, get_coord)
while 1:
    cv2.imshow(measure_win, temp_image)
    cv2.waitKey(20)
    if len(coords) >= 2:
        break
cv2.imshow(measure_win, output_image)
assert(len(coords) == 2)
assert(abs(coords[1][0] - coords[0][0]) <= 2)  # The lane line should be vertical after transformation.

dash_length_in_pixels = ((coords[1][0] - coords[0][0])**2 + (coords[1][1] - coords[0][1])**2) ** (1 / 2)
print(f"dash length {dash_length_in_pixels:.03f} pixels")

coords = []
temp_image = np.copy(output_image)
while 1:
    cv2.imshow(measure_win, temp_image)
    cv2.waitKey(20)
    if len(coords) >= 2:
        break

cv2.imshow(measure_win, output_image)
cv2.imwrite('../output/perspective_measurments.png', output_image)

assert(len(coords) == 2)
assert(abs(coords[1][1] - coords[0][1]) <= 2)  # The shortest distance between lanes should be horizontal.
lane_width_in_pixels = ((coords[1][0] - coords[0][0])**2 + (coords[1][1] - coords[0][1])**2) ** (1 / 2)
print(f"lane width {lane_width_in_pixels:.03f} pixels")

x_meter_per_pixel = lane_width_in_meters / lane_width_in_pixels
y_meter_per_pixel = dash_length_in_meters / dash_length_in_pixels
visual_length_in_meters = h * y_meter_per_pixel
visual_width_in_meters = w * x_meter_per_pixel

print(f"visual length {visual_length_in_meters:.03f} meters")
print(f"visual width {visual_width_in_meters:.03f} meters")

# Adjust transform according to expected visual range.
dst_width *= visual_width_in_meters / expected_visual_width_in_meters
dst_height *= visual_length_in_meters / expected_visual_length_in_meters
dst = np.float32([[(w - dst_width) // 2, h],
                  [(w - dst_width) // 2, h - dst_height],
                  [(w + dst_width) // 2, h - dst_height],
                  [(w + dst_width) // 2, h]])

transformation_matrix = cv2.getPerspectiveTransform(src, dst)
inv_transformation_matrix = cv2.getPerspectiveTransform(dst, src)
final_transformed_image = cv2.warpPerspective(input_image, transformation_matrix, (w, h), flags=cv2.INTER_LINEAR)

result_win = 'Final perspective'
cv2.namedWindow(result_win)
cv2.imshow(result_win, final_transformed_image)
cv2.imwrite('../output/perspective_result.png', final_transformed_image)

y_meter_per_pixel *= expected_visual_length_in_meters / visual_length_in_meters
x_meter_per_pixel *= expected_visual_width_in_meters / visual_width_in_meters
visual_length_in_meters = h * y_meter_per_pixel
visual_width_in_meters = w * x_meter_per_pixel

print("transformation matrix", transformation_matrix)
print("inv_transformation_matrix", inv_transformation_matrix)
print(f"x resolution {x_meter_per_pixel:.03f} meters/pixel")
print(f"y resolution {y_meter_per_pixel:.03f} meters/pixel")
print(f"adjusted visual length {visual_length_in_meters:.03f} meters")
print(f"adjusted visual width {visual_width_in_meters:.03f} meters")

with open('../output/perspective_transform.p', 'wb') as fid:
    output = {
        "transformation_matrix": transformation_matrix,
        "inv_transformation_matrix": inv_transformation_matrix,
        "x_meter_per_pixel": x_meter_per_pixel,
        "y_meter_per_pixel": y_meter_per_pixel,
        "visual_length_in_meters": visual_length_in_meters,
        "visual_width_in_meters": visual_width_in_meters
    }
    pickle.dump(output, fid)

cv2.waitKey()
cv2.destroyAllWindows()
