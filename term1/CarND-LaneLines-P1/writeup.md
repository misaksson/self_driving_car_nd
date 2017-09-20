**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline have four major steps:
1. Extract raw lines detection using edge detection.
2. Extract raw lines detections using color thresholding.
3. Group similar raw lines and calculate their average.
4. Average the lines in current frame with historical data

#### Raw lines from edge detection
First, I converted the images to gray scale, then I applied a Gaussian blur filter to remove some noise, then a canny filter to extract the edges. At this point, everything except the region of interest was masked out. Then I used the hough transform to detect raw line segments. The parameters for the hough transform was chosen to produce lots of small line segments, rather then just a few longer ones. This to make the detections more robust. All the small line segments are handled in a later step.

#### Raw lines from color thresholding
First, I converted the RGB images to HSV images, because I found it easier to find robust threshold values in the HSV color representation. For yellow, I used both lower and upper thresholds in all three HSV channels. But for white, it was good enough to just look at the V channel. After combining the both binary images, everything except the region of interest was masked out. Then I used morphological closing operations on the binary images to remove holes in the detected foreground objects. I then reduced the foreground objects down to their skeletons by iterating a number of morphological operations and other binary image operations. Raw lines was then finally extracted from the skeleton image by using the hough transform.

#### Group similar raw lines and find their average
The raw lines from the both extractions methods presented above was then grouped together by looking at their slopes and interception with the y-axis. I assumed here that the left and right lane would not have the same slope, and also that they would be the most frequently represented in the raw line segments. In other words, a histogram was calculated for the raw line segments slope values, and the two largest bins was assumed to belong to the left and right lines. The histogram calculation was weighted by line segments length get less sensitive to noise in the detection. After this initial grouping, some lines where removed by comparing the y-axis interception. In this I chose to go with a simpler approach and only looking at the median value, which however gave bad result in some images where there was two parallel lines with similar amount of line segments. The median value might then end up in between the parallel lines which in fact may remove all line segments. The histogram selection method would have been better also for the interception values.

To find the best line measures from current frame, the average of each group of lines from the previous step are calculated. In this, some line segments are also removed when deviating too much. The average calculation is done line by line, starting from the bottom of the image just to make sure that those line segments, which probably are the most robust ones, not will be removed due to drifting average value. The length of the lines are used as weights in this calculation to reduce the impact from short noisy lines.

#### Average line in history
In each frame, the detected lines will be added to a history record if they fit within a tolerance measure of that history instance. There are currently two hard-coded history instances, e.g. the left and right lane line. To get this started, there are an initial valid range for line slopes defined for each of the history instances. This initial valid range will also be applied if the history record for some reason becomes empty later on. When a history instance contains one more entries, new lines are only added if they are within a tolerance measure from current average. If the average history line lose track of the actual line, that is if all new lines are rejected for a few frames, then the oldest stored line will be removed until either new lines start to fit with the average again, or only an empty list remains. In this way the algorithm is somewhat robust to find lost lines. The history is currently limited to 9 entries to make the algorithm reasonable fast at responding to changes.

For each frame, the average left and right lane line is then calculated from the both history instances, and drawn as an overlay on the resulting image.


### 2. Identify potential shortcomings with your current pipeline

One shortcoming with my algorithm is when there are several significant lines in parallel, i.e. having the same slope but different y-axis interception. In the best case scenario, only one of the lines will be presented, but there may also be no detection.

Another shortcoming might be that the color thresholding, and maybe also the edge detection, performs poorly in some light conditions.

A third shortcoming is that the algorithm needs the lines to have a slope, so it does currently not work on vertical lines.


### 3. Suggest possible improvements to your pipeline

* One potential improvement could be to use a dynamic region-of-interest to achieve better fit in curves and over hills etc. The dynamic ROI could for example be based on vehicle data from CAN/flexray bus such as yaw rate, or GPS map, or image flow calculations.

* All parameters should be tuned further. I have only done some rough adjustments after visually checking the result in the example images. Some kind of auto tuning using known reference data would be nice.

* The code is by no means optimized for processing speed, and some parts of the algorithm is probably not suited for processing in a real-time embedded systems.
