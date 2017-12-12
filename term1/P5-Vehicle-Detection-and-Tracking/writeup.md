
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car.png
[image2]: ./writeup_images/car_hog_features.png
[image3]: ./writeup_images/feature_importance.png
[image4]: ./writeup_images/camera_alignment_calibration.png
[image5]: ./writeup_images/grid_levels.png
[image6]: ./writeup_images/classification_display.png
[image7]: ./writeup_images/clustering_display.png
[image8]: ./writeup_images/tracking_display.png
[image9]: ./writeup_images/output_display.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG, spatial and color histogram features, are extracted in the module ./src/feature_extractor.py. This module is used both for training and classification.

HOG features can be visualized by drawing an edge representation of the most frequent gradient in each histogram block. For a car image like this:

![alt text][image1]

HOG features using 12 gradient orientations, 8x8 pixels per cell, 2x2 cells per block and extracted from each YUV color channel, will then look like this:

![alt text][image2]

I implemented a small tool that runs exhaustive searches over different HOG parameters (and other feature extraction parameters). For each unique set of parameters, a linear SVM classifier is trained on about 15000 samples and the accuracy score is calculated on about 3000 samples. The code doing the search is available in ./tools/feature_evaluator.py

I did however not run an exhaustive search over all parameters because that would take a lot of time. Instead I tried to evaluate smaller sub-groups of parameters (more about this in the next section). Because of this, there might be better parameters than the ones I zoomed in on. There are probably also features that I use although they don't add any significant information. This is because it was hard to tell the difference between noise and actual improvements. There are for example many other parameter configurations that give roughly the same accuracy score of about 0.995 as in the parameters that I finally choose to use.

A better approach than calculating the SVC accuracy score, might be to instead use a decision tree classifier and look at the feature_importances_ attribute, which indicates the information gain of each feature. That is, if a feature has low information gain, then there is no point in extracting and train on it.

Here's a plot of the feature importances in my final classifier where I used random forest and trained it on all extracted training data (about 370k samples). It's interesting that 80% of the feature importance is accumulated from less then 3% of the features.

![alt text][image3]


#### 2. Explain how you settled on your final choice of HOG parameters.

##### Initial HOG parameter search
Using the ./tools/features_evaluator.py, I started of by only looking at HOG parameters in the HSV color space, doing a sparse search ending up at best results for 12 orients, 8 pix per cell and 3 cells.

##### Color histogram and spatial features
Then I evaluated if color histograms and/or spatial features added any information, and it was a positive on both. In the next step however, when refining the size of spatial features, it was found that the improvement provided by spatial feature actually was just noise in the measurements, concluding that spatial features added no significant value. The color histogram features does however seem to add some non-negligible performance improvement, and best result was found for 32 bins.

##### Color spaces
Next I evaluated different color spaces, including HSV, BGR, LUV, HLS, YUV, YCrCb. This is something I should have done earlier, before making conclusions on the value of different features when using a another colorspace. I got best result when using the YUV color space.

##### Final tuning of HOG parameters
Now I refined the seach for good HOG parameters, and also evaluated if all color-channels added some value for HOG, which they did. The result was however almost as good when only using the Y- and V- channels.

The final feature extraction parameters that I used:

* HOG:
  * 12 orients
  * 8x8 pixels per cell
  * 3x3 cells per block
  * All channels in the YUV colorspace.
* Color histogram with 32 bins.

The raw result log from the parameter search is provided in ./Output/feature_evaluator.txt

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

##### Training
The module ./src/trainer.py implements
1. Loading of training data (trainer.py line 54-166)
  * Selection of training set(s).
  * Equalization of number of samples that are labeled vehicle resp. background.
  * Drops an configurable amount of randomly selected samples from time-series data. This is mainly to reduce the amount of training data in order to make trainings faster. Consecutive samples in the time-series data are however very similar and might influence the training too much, almost like if oversampling the single examples.
2. Splitting into train/test sets. (trainer.py line 167-210)
  * For time-series, this is done before shuffling the data to avoid getting similar samples in both the training and test-set.
  * The training set with single examples are just shuffled and split.
3. Feature extraction (trainer.py line 212-246) is implemented by calling the same functions in feature_extractor.py, as used in previous section when evaluating the features. The features are then optionally scaled, but this is not necessary when using a decision tree classifier such as random forest. The feature scaling (for SVM) was problematic and used lots of memory. To be able to fit ~50k samples I had to implement a workaround that buffers the feature vectors to HDD when repacking them for the scaler. This workaround wasn't very nice so I removed it when switching to Random Forest classifier.
4. Training and validation (trainer.py line 248-280)
  * This expects an initialized classifier that implements the sklearn-API with fit(), score() etc. The application use predict_proba() so if implementing a SVM classifier, then set the probability argument when initializing the classifier.
  * If no classifier is given, then it defaults do RandomForest using the same arguments as I have used. The application will train a classifier by this method the first time it's launched, or whenever the pickle file classifier.p is missing. The next run, it just loads the classifier from the pickle file.

##### Classifier evaluator
To search for best classifier parameters, I made a tool ./tools/classifier_evaluator.py which is implemented in the same way as the feature evaluator tool, and maybe somewhat similar to the sklearn GridSearchCV. I didn't use the GridSearchCV since I need to control the train/test split for the time-series data, and I wasn't sure how to do that with GridSearchCV (maybe possible with the cv parameter?).

##### Training data set
###### Annotated data
Extra training data was extracted from the annotated sequences available here:
https://github.com/udacity/self-driving-car/tree/master/annotations

Both data sets was used, resulting in nearly 400k samples in total. The samples extracted from the annotated data were selected based on the search window grid generator used by the detection pipeline (described in next section) in an attempt to get as realistic samples as possible for the training. The same amount of vehicle and non-vehicle samples were extracted, where the later also was equalized among the grid levels. The tool used to do this is available here: ./tools/extract_training_data.py

###### Hard negative mining
In an attempt to remove false-postives in the project-video, I extracted some false-postives and used them as non-vehicle samples in the training. I think however this data had too little influence on the training, and I would either need to extract more samples or oversample current data to give it some impact. Very similar examples are anyway still present in the classified output by large quantities.
The tool I implemented to sort out negative examples are available here: ./tools/label_detections.py

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

##### Search windows positioning in real world coordinates
Search windows are provided by the ./src/grid_generator.py module. There are multiple grid levels (scales) in the grid, and the positioning and size of each grid level is defined in world-coordinates, e.g. distance in front of the vehicle, width and height of each grid level. The size of each search window is also defined in world-coordinates, by giving an expected width and height of a vehicle.

##### Camera parameters
To be able to define the grid in real world coordinates and calculate the coresponding search window positions in the camera image, I had to find some camera parameters.

###### Camera angular field-of-view
Information about the camera was found at Udacity camera-mount github repo. In the lens specification, the horizontal field-of-view is 54.13 degrees and the vertical 42.01 degrees.

http://www.fujifilmusa.com/products/optical_devices/machine-vision/1-15-mp/cf125ha-1/

###### Camera alignment calibration
To be able to positioning the grid, the camera position and attitude is also needed, in particular the pitch-angle and the height above ground (z-coordinate). To measure this, I utilized some information from the Advanced-Lane-Lines-project about the lanes width and dash lines lengths, such that a special grid could be defined as a corridor in front of the vehicle having the same width as a lane, and the grid levels positioned at the beginning and end of each dash line. By drawing the outlines of this grid on the camera image, it was possible to tune the pitch and z-coordinates such that the result looks reasonable OK for the project_video.

![alt text][image4]

##### The grid definition
The final grid is defined as a conic search corridor such that the search area is wider further away. This mainly because the positioning of the grids are more uncertain further away (no path prediction available), and it's also of more interest to find objects not directly at the path at some distance since those objects have longer time to move until the ego vehicle will get there.

In total, there are 7 grid levels, of which all but the first is positioned 20 meters apart, ranging from 15 to 120 meters. The horizontal search area range from 28 meters near the vehicle to 70 meters. The height is also conic, 7.4 to 10 meters, which is positioned such that 1/4 is below the road surface.

The search window is set to 3 meters wide and high, as an compromise between cars and trucks. Note this does not need to be spot on to get detections, although it might be good to process other sizes as well if there is processing time available.

An overlap of 0.75 was used both in vertical and horizontal direction, which seems to work OK with the classifier and tracker I have implemented. About the same tolerance is used when extracting patches from the extra annotated data.

The final grid looks like this with a single search window drawn in bottom right corner of each grid level boundary. Note that it seems reasonable that the blue grid-level will be able to pick up the black car, and red grid-level the white car.

![alt text][image5]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here's 6 examples (extracted from the project_video.mp4), showing all steps in the detection and tracking pipeline. Click on images to expand to full size.

##### Input
<table style="width:100%">
  <tr>
    <th>Frame 326</th>
    <th>Frame 446</th>
    <th>Frame 555</th>
    <th>Frame 625</th>
    <th>Frame 764</th>
    <th>Frame 980</th>
  </tr>
  <tr>
    <td><a href="./writeup_images/input_frame326.png" target="_blank"><img src="./writeup_images/input_frame326.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/input_frame446.png" target="_blank"><img src="./writeup_images/input_frame446.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/input_frame555.png" target="_blank"><img src="./writeup_images/input_frame555.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/input_frame625.png" target="_blank"><img src="./writeup_images/input_frame625.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/input_frame764.png" target="_blank"><img src="./writeup_images/input_frame764.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/input_frame980.png" target="_blank"><img src="./writeup_images/input_frame980.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
</table>

##### Classification

There are two thresholds applied on the classified objects probability value. The objects passing the lower threshold 0.6 are used in clustering (heatmap), and the objects passing the upper threshold 0.75 are used to provide new observations to the tracker. The probability of an object is indicated by the color of the bounding box, see the colorbar for mapping to probability.

<table style="width:100%">
  <tr>
    <td><a href="./writeup_images/classification_frame326_th60.png" target="_blank"><img src="./writeup_images/classification_frame326_th60.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame446_th60.png" target="_blank"><img src="./writeup_images/classification_frame446_th60.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame555_th60.png" target="_blank"><img src="./writeup_images/classification_frame555_th60.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame625_th60.png" target="_blank"><img src="./writeup_images/classification_frame625_th60.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame764_th60.png" target="_blank"><img src="./writeup_images/classification_frame764_th60.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame980_th60.png" target="_blank"><img src="./writeup_images/classification_frame980_th60.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
  <tr>
    <td><a href="./writeup_images/classification_frame326_th75.png" target="_blank"><img src="./writeup_images/classification_frame326_th75.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame446_th75.png" target="_blank"><img src="./writeup_images/classification_frame446_th75.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame555_th75.png" target="_blank"><img src="./writeup_images/classification_frame555_th75.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame625_th75.png" target="_blank"><img src="./writeup_images/classification_frame625_th75.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame764_th75.png" target="_blank"><img src="./writeup_images/classification_frame764_th75.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/classification_frame980_th75.png" target="_blank"><img src="./writeup_images/classification_frame980_th75.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
</table>


##### Clustering

The classifier output lots of detections of the same object, and also lots of false detections. The cluster module group these detections into one or a few detections that then are provided as potential new candidates to the tracker module. In this process, some of the false detections will not fit in, and be removed.

My cluster implementation use a heatmap image, where each pixels intensity indicates how likely it is that an object are located on that position in the image. The heatmap is built up by accumulating the objects confidence score into an image that at first is initialized to all zeros. That is, all pixels within an objects bounding box is incremented by that objects confidence score, which is a value based on the probability, but mapped using an exponential function such that probabilities close to 1.0 weigh more than several detections that barely pass the threshold of 0.6. Such an heatmap is generated for each frame, and then heatmaps from a few previous frames added together to further improve the stability in the detections. I found that 3 frames was enough to provide some stability, and still not lose the ability to catch the fast moving oncoming vehicles. To produce output objects, the heatmap is typically thresholded and labeled, such that the extreme coordinates of each object can be extracted from the labeled image. In my implementation, these objects did however get strange and instable aspect ratio, making them unsuitable to match with previous detections in the tracker. Instead choose to only use these detections as indications of where there are good objects, and instead use the average position and size of the 3 objects with highest probability in that position.

In the heatmaps below, the threshold applied before clustering is at confidence score 75, see the colorbar to understand what will be removed by the threshold. The clustered objects are also drawn as bounding boxes.

<table style="width:100%">
  <tr>
    <td><a href="./writeup_images/cluster_frame326.png" target="_blank"><img src="./writeup_images/cluster_frame326.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/cluster_frame446.png" target="_blank"><img src="./writeup_images/cluster_frame446.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/cluster_frame555.png" target="_blank"><img src="./writeup_images/cluster_frame555.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/cluster_frame625.png" target="_blank"><img src="./writeup_images/cluster_frame625.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/cluster_frame764.png" target="_blank"><img src="./writeup_images/cluster_frame764.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/cluster_frame980.png" target="_blank"><img src="./writeup_images/cluster_frame980.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
</table>

##### Tracking

The tracker takes input objects both from the cluster and directly from the classifier, where the later is the high probability objects that passed the upper threshold of 0.75 probability. The clustered objects are only used as new track candidates. A new track is added when there are no ongoing track matching the candidate. Objects from the classifier on the other hand are only used as observations for already ongoing tracks. They may adjust the position and size of a track and also increase the confidence score if they match the track good enough. The amount by which an observation is influencing the track depends on the confidence of both the observation and the track. A new track, that not yet has reached a high confidence will typically be adjusted more than an old stable track. If there are no observation of a track in a frame, then the confidence score might be decreased. That is when there is no optical flow measurement of high enough confidence, such that the track is allowed to keep it's confidence. Optical flow are of course also used to update the position and size of a track, with or without other any observations. If there are no observations and no measurements, than the track falls back to predictions based on delta changes in previous frame. When this happens, the confidence is slightly decreased such that it will eventually be removed. There are two confidence thresholds, one for output objects, and the other for dropping tracks. In this way the tracker can silently continue to follow potential objects even if not confident enough to provide the track in the output. There are also an age threshold preventing very young tracks from showing up in the output, forcing tracks to stabilize (or be removed) before being provided in the output.

I think this gives an overview of the tracker. There are lots of details in the implementation, more or less messy, because I was experimenting a lot with different thresholds and parameters.

In the images below, the confidence of a track is indicated by the color of the bounding box. Tracks must have a confidence of at least 10 to be provided in the output. The black boxes are observations from the classifier that matched a track. The colored dots within each object is tracked using optical flow.

<table style="width:100%">
  <tr>
    <td><a href="./writeup_images/tracking_frame326.png" target="_blank"><img src="./writeup_images/tracking_frame326.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/tracking_frame446.png" target="_blank"><img src="./writeup_images/tracking_frame446.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/tracking_frame555.png" target="_blank"><img src="./writeup_images/tracking_frame555.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/tracking_frame625.png" target="_blank"><img src="./writeup_images/tracking_frame625.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/tracking_frame764.png" target="_blank"><img src="./writeup_images/tracking_frame764.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/tracking_frame980.png" target="_blank"><img src="./writeup_images/tracking_frame980.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
</table>

##### Final output

The tracked objects with enough confidence and age are then provided as the final output from the pipeline and displayed like this.

<table style="width:100%">
  <tr>
    <td><a href="./writeup_images/output_frame326.png" target="_blank"><img src="./writeup_images/output_frame326.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/output_frame446.png" target="_blank"><img src="./writeup_images/output_frame446.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/output_frame555.png" target="_blank"><img src="./writeup_images/output_frame555.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/output_frame625.png" target="_blank"><img src="./writeup_images/output_frame625.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/output_frame764.png" target="_blank"><img src="./writeup_images/output_frame764.png" alt="" width="180" height="90" border="0" /></a></td>
    <td><a href="./writeup_images/output_frame980.png" target="_blank"><img src="./writeup_images/output_frame980.png" alt="" width="180" height="90" border="0" /></a></td>
  </tr>
</table>

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a Youtube video showing the final result:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=UKeJ6Mod44k" target="_blank"><img src="http://img.youtube.com/vi/UKeJ6Mod44k/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

False positives are being removed both by the cluster and the tracker modules, and the same modules are also combining overlapping bounding boxes. The cluster is implemented in ./src/cluster.py and the tracker in ./src/tracker.py. Both modules are explained in the pipeline demo in previous section.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was struggling a lot with the classification, and still think it's the weakest part of the pipeline. The bad classification results are to some degree covered for by the cluster and the tracker, but it was hard to tune the pipeline such that it produced the final result as seen in the video.

If I were going to pursue this project further, then I would start by trying to improve the selected features. The scatter plot presented above, that shows the importance of each feature used in current training suggests that there are room for improvements in this area. Maybe evaluate other types of features such as Viola-Jones, SIFT and/or SURF.
