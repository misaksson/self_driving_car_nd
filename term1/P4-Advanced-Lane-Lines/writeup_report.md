**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/chessboard_undistorted.png "Undistorted"
[image2]: ./writeup_images/1_input.png "Input image"
[image3]: ./writeup_images/2_undistorted.png "Undistorted"
[image4]: ./writeup_images/3_binary.png "Binary image"
[image5]: ./writeup_images/threshold_tuning_tool.png "Threshold tuning tool"
[image6]: ./writeup_images/perspective_source.png "Perspective source points"
[image7]: ./writeup_images/perspective_measurments.png "Perspective measurements"
[image8]: ./writeup_images/perspective_result.png "Perspective result"
[image9]: ./writeup_images/4_perspective.png "Perspective binary image"
[image10]: ./writeup_images/5_0_line_base.png "Line base detection"
[image11]: ./writeup_images/5_1_initial_line_fit.png "Line sliding windows"
[image12]: ./writeup_images/5_2_line_fit.png "Line refine"
[image13]: ./writeup_images/6_final_result.png "Final result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in ./src/camera_calibration.py. Calling the module directly will show demo figures of chessboard
corner detection and distorted vs undistorted images.

There are 20 chessboard images provided for this project, which are captured by the same camera as is mounted in the vehicle. Each image show a view of a chessboard paper that is mounted on a wall. This provides for a fixed (x, y) plane at z=0, such that the object points (world coordinates) are the same for each calibration image. The corresponding image points are provided by cv2.findChessboardCorners. I did however use a varying number of corners to search for in the chessboard to be able to use as many of the calibration images as possible. If the cv2.findChessboardCorners fails for the expected 9x6 corners, then a few more variants are tried. This made it possible to use all but one image for calibration (not sure why calibration4.jpg failed), but since the number of object points then not are known in advance, they are created individually after each successful chessboard detection.

Then cv2.calibrateCamera was used to calculate camera calibration and distortion coefficients. The cv2.undistort can then finally remove radial and tangential distortion from an image as seen in this example:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

An input image like this:
![alt text][image2]

was corrected for distortion by calling the Calibrate.undistort method in the camera_calibration module. It applies previously calculated calibration and distortion coefficients to the image using the cv2.undistort function. The result is seen here:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've implemented filters for color thresholding of RGB and HSV image channels and several different gradient thresholds based on the sobel kernel. This is used to produce binary images such as this:
![alt text][image4]

This code is located in ./src/extractor.py, and is built up using a class structure that makes it easy to configure and combine different filters into any hierarchy.

I also put some effort into developing a tool for threshold tuning, that automatically creates a grid of windows, one for each configured filter and combination of filters. Windows with filters also contains sliders for threshold tuning. The tool plays a video sequence and the input window have a slider for frame selection. The resulting thresholds are stored when shutting down and then loaded when opening the tool again, or when running the actual lane finding pipeline. The code for this tool is in ./tools/tune_thresholds.py and here's a screen-shot:
![alt text][image5]

In the end there wasn't that much time to play around with the tuning so I ended up using only color thresholds, mainly relying on the red-channel which provides good results for the project_video.mp4, except in the very brightest spots, where I instead used a combination of hue, lightness and saturation. The lightness filter was tuned such that when combined with saturation and hue, this filter only provided output for bright spots. I also played around some with the sobel filters trying to get detections also in the shadows for the more challenging videos, but without good result. A similar method to the one applied for bright spots might however be good also for shadows, e.g. using the lightness filter to only apply sobel filters when in shadow.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I implemented an interactive tool for calculation of perspective transform, available in ./tools/find_perspective_transform.py. The first step (code lines 55-70) when running this tool is that the user selects source points by drawing four lines like this:
![alt text][image6]

The image is then transformed to show the selected perspective (code lines 72-82), for now just using some hard-coded destination points. The next step is to measure the lane-lines dash length and the lane width in pixels, within the transformed image (code lines 84-116). These values are then used to calculate pixel resolution in real-world coordinates (meter per pixel), in x resp y-direction (code lines 118-119). This is possible since we know that the dash length is 3 meters and the lane width 3.7 meters. The red-lines in the image below show such an measurement:
![alt text][image7]

The final step is to repeat the perspective transform, but this time using destinations points that will produce a view looking 30 meters ahead and 5 meters to each side of the vehicle (code lines 120-136). The final perspective then looks like this:
![alt text][image8]

Another small tool (./tools/apply_perspective_transform.py) was implemented to produce a video where the transformation is applied on an image and the result is displayed side-by-side with the original image. This was used to verify that the transformation works as expected, e.g. by checking that the lane lines typically are in parallel both on strait road and in curves.

The perspective transformation matrix is then applied in the lane finding pipeline on the extracted binary image (./src/main.py line 173-176) producing an image like this:
![alt text][image9]

The inverse transformation matrix is used to draw the detected lane as overlay in the final output image (./src/main.py line 49-70), see example of this further down.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane pixels are identified and fitted to a polynomial in ./src/detector.py. There are two methods implemented, one for finding lines when there are no prior detections (code line 37-148), and one method to refine previous detection for current frame (code line 150-249).

##### Search lines
This method is applied for the first initial detection, and also when the algorithm has lost track of the lane line.
The first step here is to identify where the lane line starts, e.g. at what x-value it intercepts the image bottom. This is done by first calculating a histogram as the vertical sum of lower half of the image (code line 58-59). It will look something like this:
![alt text][image10]

The highest peaks on left and right side of the histogram are then assumed to belong to the lane lines. For each line, search windows are then used to trace the line upwards in the warped binary image. This by sliding the search window for each iteration to the mean pixel x-coordinate in previous search window. The result may look like this:
![alt text][image11]

Pixels within the search windows are then used to fit a second degree polynomial (code line 134).

##### Refine lines
This method is applied whenever there are previous knowledge about the lane lines. Instead of search windows, a boundary is selected close to previously detected line (green curved lines). If the number of extracted pixels within the boundary are above a threshold, then the line detection is considered valid and the fitted polynomial are added to a history record, from which the average polynomial coefficients will be calculated and used as line detection for this frame (code line 224-248). On the other hand, if there aren't enough extracted pixels within the boundary, then this frame will not provide any new knowledge about the line position, and eventually, if this happens often then the line will be lost and the initial search method applied again. In the image below, the red and blue pixels are within left resp. right search boundary. The yellow line is the fitted polynomial for this frame and cyan line is the average polynomial, which also are what's considered to be the detection.
![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

##### Curvature radius
Curvature radius is calculated in ./src/detector.py line 250-280. The line curvature radius is calculated just in front of the vehicle. The previously calculated pixel resolution in meters is used to fit a second degree polynomial in real-world coordinates. The line radius is calculated near to the vehicle using this formula:
R = (1 + f'(y)^2)^(3/2) / abs(f"(y)), where a second degree polynomial f(y) = Ay^2 + By + C have f'(y)= 2Ay + B and f"(y) = 2A, resulting in R = ((1 + (2Ay + B))^2)^(3/2) / abs(2A).

Finally, I choose to set negative value on the curvature to indicate when it's turning left, that is when f"(y) = 2 * A is negative.

##### Vehicle position in lane
This is calculated in ./src/detector.py line 282-295 and 313-317. The center of the transformed binary image is assumed to be the center of the vehicle, then the distance in pixels to left and right polynomial is calculated where the line intercepts the bottom of the transformed image, e.g. just in front of the vehicle. The distance in pixel is then converted to meters by using previously calculated pixel resolution (meters per pixel). Since we know the lane width, the both distance calculations actually provides two measurements of the vehicles position in the lane, such that the average value can be calculated.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane detection result is transformed to (undistored) camera image perspective using the inverse transformation matrix, and plotted as an filled green polygon that are blended into the image at ./src/main.py line 49-70. The lane boundary is calculated at ./src/detector.py line 321-325. Here's an example of what the final output image looks like:
![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a Youtube video showing the final result:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=vdv0Lh9MBVE" target="_blank"><img src="http://img.youtube.com/vi/vdv0Lh9MBVE/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The major problem with current algorithm is in the binary image extraction of pixel coordinates that are likely to belong to lane lines. Currently this only works reasonable OK during good light conditions, such as in the project_video.mp4. To be able to handle other light conditions, road pavements, shadows, etc. a more robust extraction algorithm is needed.
* Skip line detections that are deviating too much from previous detections.
* Improve robustness of lane detection by combining the result from both lines:
  * Use the average curvature of the both lane lines as final steering output.
  * Use an offset to the left lane line in order to improve the search boundary of the right lane line, and vice versa.
  * Validate left and right lane lines with each other.
* There is some problem with the curvature calculation, but I was not able to find the error. It looks like it's a factor 3 smaller than the expected curvature in the beginning of project_video.mp4.
