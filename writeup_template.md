
[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/banner.png "Banner"
[image9]: ./output_images/dataset_exploration.png "dataset_exploration"
[image10]: ./output_images/hog.png "hog"
[image11]: ./output_images/window.png "window"
[image12]: ./output_images/sliding_window_roi.png "sliding_window_roi"
[image13]: ./output_images/multiple_detection1.png "multiple_detection1"
[image14]: ./output_images/multiple_detection2.png "multiple_detection2"
[image15]: ./output_images/heatmap_without_threshold.png "heatmap_without_threshold"
[image16]: ./output_images/heatmap_with_threshold.png "heatmap_with_threshold"
[image17]: ./output_images/heatmap_grayscale.png "heatmap_grayscale"
[image18]: ./output_images/object_detection_final.png "object_detection_final"
[image19]: ./output_images/output1.gif "output1"
[image20]: ./output_images/output2.gif "output2"
[video1]: ./project_video.mp4

# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![banner][image8]

## Goal - To write a software pipeline to detect vehicles in a video.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### 1. Data Exploration
#### Code - Section 115 ``` vehicle_detection_notebook.ipynb```
- Data was extracted from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
- Ensured that the number of positive and negative samples are of approximately the same count so that the learning is not biased.

 **Vehicle Size**                     |  **Non-Vehicle Size** 
 :-------------------------:|:-------------------------:
 8793 |  8968
 

- I started by reading in all the `vehicle` and `non-vehicle` images.  
- Here is an example of `vehicle` and `non-vehicle` classes along with the other statistics:
![alt text][image9]

### 2. Color Spaces and Histogram of Oriented Gradients
#### Code - Section 116 & 117 ``` vehicle_detection_notebook.ipynb```
- Explored different color spaces RGB, HSV, HLS, LAB, YCRB and different `skimage.hog()` and parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

![alt text][image10]


### 3. Sliding Window and Region of Interest
#### Code - Section 36 & 151  ``` vehicle_detection_notebook.ipynb```
- Implemented the Sliding window function which will be used extract patches of images.
- **Reason for scaling in Sliding window**: We know that the car would look smaller at a farther distance and bigger when it is closer.
- **Reason for Region of interest**: Cars are not going to be present in the sky. Having a ROI can minimze the computations needed.


 **Without ROI**                     |  **With ROI** 
 :-------------------------:|:-------------------------:
![SlidingWindow][image11] |  ![SlidingWindow-ROI][image12]

 
### 4. Final Configuration parameters after experimentation
- After taking educated guesses(also tried to use parameter tuning) arrived with the given set of parameters that work fine for detecting and separating cars from non-cars.
- Also, the HOG works fine on the following Color spaces - LUV, YUV and YCrCb

 **Orientations**                     |  **Pixels per cell** |  **Cells per block** |  **Feature Vector Size** 
 :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
11  |  16 |  2 |  1188


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

