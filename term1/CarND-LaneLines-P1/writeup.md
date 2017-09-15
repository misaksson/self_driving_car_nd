**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline have three major steps:
1. Extract raw lines detection using edge detection.
2. Extract raw lines detections using color thresholding.
3. Group similar raw lines and calculate their average.

#### Raw lines from edge detection
First, I converted the images to gray scale, then I applied a Gaussian blur filter to remove some noise, then a canny filter to extract the edges. At this point, everything except the region of interest was masked out. Then I used the hough transform to detect raw line segments. The parameters for the hough transform was chosen to produce lots of small line segments, rather then just a few longer ones. This to make the detections more robust. All the small line segments are handled in a later step.

#### Raw lines from color thresholding
First, I converted the RGB images to HSV images, because I found it easier to find robust threshold values in the HSV color representation. For yellow, I used both lower and upper thresholds in all three HSV channels. But for white, it was good enough to just look at the V channel. After combining the both binary images, everything except the region of interest was masked out. Then I used morphological closing operations on the binary images to remove holes in the detected foreground objects. I then reduced the foreground objects down to their skeletons by iterating a number of morphological operations and other binary image operations. Raw lines was then finally extracted from the skeleton image by using the hough transform.

#### Group similar raw lines and find their average
The raw lines from the both extractions methods presented above was then grouped together by looking at their slopes and interception with the y-axis. I assumed here that the left and right lane would not have the same slope, and also that they would be the most frequently represented in the raw line segments. In other words, a histogram was calculated for the raw line segments slope values, and the two largest bins was assumed to belong to the left and right lines. The histogram calculation was weighted by line segments length get less sensitive to noise in the detection. After this initial grouping, some lines where removed by comparing the y-axis interception. In this I chose to go with a simpler approach and only looking at the median value, which however gave bad result in some images where there was two parallel lines with similar amount of line segments. The median value might then end up in between the parallel lines which removed all line segments. The histogram selection method would have been better also for the interception values.

To find the final left and right lane line, the average of each group of lines from the previous step was calculated. In this, some line segments was also removed when deviating too much. The average calculation was done line by line, starting from the bottom of the image just to make sure that those line segments, which probably are the most robust ones, not would be removed due to the too the much deviation criteria. During the average calculation, the extreme line coordinates are also extracted and used when calculating the final line to be presented in the image.


### 2. Identify potential shortcomings with your current pipeline


One shortcoming with my algorithm is when there are several significant lines in parallel, i.e. having the same slope but different y-axis interception. In the best case scenario, only one of the lines will be presented, but there may also be no detection.

Another shortcoming might be that the color thresholding, and maybe also the edge detection, performs poorly in some light conditions.

A third shortcoming is that the algorithm needs the lines to have a slope, so it does currently not work on vertical lines.


### 3. Suggest possible improvements to your pipeline

* A potential improvement would be to use history from previous frames e.g. as starting point for the line average calculation.

* I should probably have extrapolated the detected lines down to the bottom of the image, and maybe also up to some fixed length in order to improve the visual impression.

* Another potential improvement could be to use a dynamic region-of-interest to achieve better fit in curves and over hills etc. The dynamic ROI could for example be based on vehicle data from CAN/flexray bus such as yaw rate, or GPS map, or image flow calculations.

* All parameters should be tuned further. I have only done some rough adjustments after visually checking the result in the example images. Some kind of auto tuning using known reference data would be nice.

* The code is by no means optimized for processing speed, and some parts of the algorithm is probably not suited for processing in a real-time embedded systems.
