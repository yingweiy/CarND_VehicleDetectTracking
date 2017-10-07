**Udacity CarND Term 1 Project 5**
# Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[carhog]: ./output_images/CarHOG.jpg
[noncarhog]: ./output_images/NonCarHOG.jpg
[carhogs]: ./output_images/CarHOG_L.jpg
[noncarhogs]: ./output_images/NonCarHOG_L.jpg

[car]: ./car.png
[noncar]: ./noncar.png
[test1]: ./output_images/test1.jpg
[test2]: ./output_images/test2.jpg
[test3]: ./output_images/test3.jpg
[test4]: ./output_images/test4.jpg
[test5]: ./output_images/test5.jpg
[test6]: ./output_images/test6.jpg
[heat1]: ./output_images/heat_test1.jpg
[heat2]: ./output_images/heat_test2.jpg
[heat3]: ./output_images/heat_test3.jpg
[heat4]: ./output_images/heat_test4.jpg
[heat5]: ./output_images/heat_test5.jpg
[heat6]: ./output_images/heat_test6.jpg
[th1]: ./output_images/th_test1.jpg
[th2]: ./output_images/th_test2.jpg
[th3]: ./output_images/th_test3.jpg
[th4]: ./output_images/th_test4.jpg
[th5]: ./output_images/th_test5.jpg
[th6]: ./output_images/th_test6.jpg
[label1]: ./output_images/label_test1.jpg
[label2]: ./output_images/label_test2.jpg
[label3]: ./output_images/label_test3.jpg
[label4]: ./output_images/label_test4.jpg
[label5]: ./output_images/label_test5.jpg
[label6]: ./output_images/label_test6.jpg

[video]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

##### Program Files:

* `CarDetection.ipynb`: the main program in ipynb format
* `CarDetection.pdf`: the main program in PDF format
* `lessonFunctions.py`: supported functions


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the lines 8 through 25 of the file called `lessonFunctions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes :

![alt text][car]
![alt text][noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the saturation channel (from HLS encoded image) and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

Car and their HOG:

![alt text][carhogs]

NonCar and their HOG:

![alt text][noncarhogs]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the best configuration is as follows:
* input format: gray image
* orientations: 9
* pixcels per cell: 8
* cells per block: 1

The examples of car and their HOG:

![alt text][carhog]

NonCar and their HOG:

![alt text][noncarhog]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of color bin and histogram on HLS encoded image in together with HOG features on gray image.

##### Features
The configuration of to generate these features:

```python
color_space = 'HLS' 
orient = 9
pix_per_cell = 8
cell_per_block = 1
hog_channel = 'GRAY' 
hist_bins = 16
spatial_size=(8, 8)
spatial_feat = True
hist_feat = True
hog_feat = True
```

These paramters are used to call the features extraction function:

```python
car_features = extract_features(cars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins = hist_bins,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins = hist_bins,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
```

##### SVC Classification

By experiments, I found with the following paramters, it achieves the best testing accuracy of 99.27%.

```python
svc = SVC(C=1e8, gamma=4e-4)
```


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using gray HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heat1]

![alt text][heat2]

![alt text][heat3]

![alt text][heat4]

![alt text][heat5]

![alt text][heat6]

### Here is the output of threshold and `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][th1]

![alt text][th2]

![alt text][th3]

![alt text][th4]

![alt text][th5]

![alt text][th6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][label1]
![alt text][label2]
![alt text][label3]
![alt text][label4]
![alt text][label5]
![alt text][label6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The use of SVC with RBF kernel greatly improves the results. The classifier is very sensitive to the C and gamma paramters.
* The use of color bin and histogram together with HOG can improve the overall accuracy.
* The HLS encoding outperform other encodings in my experiements. 

The problems that I can observe from the resulting video:
* The bonding box looks kind of shaky.
* There is still false alarm, such as in the lawn. 

To improve it, I think a deep neural network approach may be better than the feature engineering approach.
  

