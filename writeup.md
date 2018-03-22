
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

**Result 1**                     |  **Result 2** 
 :-------------------------:|:-------------------------:
![alt text][image19] |  ![alt text][image20]

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
- The HOG extractor extracts meaningful features of a image. 
- It captures the common aspects of cars, not the specifics of it.
- It is the same as humans (at the first glance), we locate the car, not the model, the tires, or other small details.
- It divides an image into several pieces. For each piece, it calculates the gradient of variation in a given number of orientations. 
- The idea is that the HOG captures the essence of original image.

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
#### Code - Section 158 ``` vehicle_detection_notebook.ipynb```

- After taking educated guesses(also tried to use parameter tuning) arrived with the given set of parameters that work fine for detecting and separating cars from non-cars.
- Also, HOG works fine on the following Color spaces - LUV, YUV and YCrCb
- Experiment 1: Changing the ROI
- Experiment 2: Changing the Scaling
- Experiment 3: Changes in YUV color space

 **Orientations**                     |  **Pixels per cell** |  **Cells per block** |  **Feature Vector Size** 
 :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
11  |  16 |  2 |  1188


#### 5. SVM Classfier
#### Code - Section 131 ``` vehicle_detection_notebook.ipynb```
- These data are separated in training (80%) and validation sets (20%)
- In this case, I used a Support Vector Machine Classifier (SVC), with linear kernel.
- A SVM finds a line that better divides two sets. 
- Referal: http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
- The car can appear in different sizes. 
- Then, we apply different windows sizes over the image. 
- Spatial bins not used as it was considered unnecessary.

**Accuracy**                     |
 :-------------------------:
98.17%  |


### 6.Sliding Window Search
#### Code - Section 151 ``` vehicle_detection_notebook.ipynb```

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

- Adapted the method ```find_cars``` from the udacity materials. 
- The method combines HOG feature extraction with a sliding window search.
- Instead of calculating feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image.
- The full-image features are subsampled according to the size of the window.
- Then the respective portion is fed to the classifier. 
- Prediction on the HOG features for each window is performed and a list of rectangle objects are returned if there is a match.

### 7. Multiple Detections

 **Multiple Detections 1**                     |  **Multiple Detections 2** 
 :-------------------------:|:-------------------------:
![alt text][image13] |  ![alt text][image14]

---
### 8. False Positives and Combining Overlapping regions with Heatmap

- Heatmap are necessary to find the overlapping regions.
- Overlapping regions can be used to measure the confidence.
- Confidence can be thresholded to remove false positives.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


**Heatmap without Thresholding**  |  **Heatmap with Thresholding** |  **Heat Map GrayScale** 
 :-------------------------:|:-------------------------:|:-------------------------:|
![alt text][image15]  |  ![alt text][image16] |  ![alt text][image17]

- Combining the overlapping region and forming a single bounding box per car.


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

**Object Detection Final**                     |
 :-------------------------:
![alt text][image18] |

**Result 1**                     |  **Result 2** 
 :-------------------------:|:-------------------------:
![alt text][image19] |  ![alt text][image20]

### Video Implementation
- Here's a [link to my video result](./output.mp4)
- Works decently well on the lane detection video output too [link to my video result](./output_with_lane.mp4)
---

### Discussion
- There are a lot of false positives.
- When we try to increase the number of windows, we compromise on the realtime speed requirement.
- If we use the previous frame as an approximate position of the car in the next frame, we loose upcoming traffic which changes it position drastically.
- Hand tunining the configuration parameters works fine but is not a scalable solution. E.g If we are supposed to detect bikes using the same code, it would **not** work.
- Following Neural Networks would do a better job of detection without much tuning:
- Single Shot Multibox Detector (SSD) with MobileNets
- SSD with Inception V2
- Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101
- Faster RCNN with Resnet 101
- Faster RCNN with Inception Resnet v2

Note: In most of the cases above, we only need to train the penultimate layer.

### References
- https://medium.com/@jeremyeshannon
- http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
- https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9
- Udacity nanodegree Self_Driving_Car_ND
- https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
