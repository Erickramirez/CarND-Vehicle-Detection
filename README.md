**Vehicle Detection Project**

The steps of this project are the following:

* Apply a distortion correction to raw images
* Perform color space change from [RGB](https://en.wikipedia.org/wiki/RGB_color_model) to [YCbCr](https://en.wikipedia.org/wiki/YCbCr)
* Perform a Histogram of Oriented Gradients (HOG) feature extraction  
* Append binned color features, as well as histograms of color, to the HOG feature vector. All this has been normalized.
* Randomize a selection for training and testing sets.
* Apply classifier Linear SVM classifier with training and testing sets.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Create vehicule detection based on the heat map.

[//]: # (Image References)
[image1]: ./examples/data_colection.png
[image2]: ./examples/YCrCb.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


## Image Classification (Car or Not car)
The code for this step is contained in the IPython notebook located in "./Vehicle-detection-training.ipynb"

####1. Collecting the Data
the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train my classifier. This is the amount of data used: ´cars: 8789 notcars: 8968´. Note: in this case I didn't use any image augmentation technique.  These are some examples of the data, in this case there are only 2 classes:
![alt text][image1]

####2. Perform color space change from [RGB](https://en.wikipedia.org/wiki/RGB_color_model) to [YCrCb](https://en.wikipedia.org/wiki/YCbCr)
I explored different color spaces and YCbCr has the best performance for the training (For my testing). 
Y is the luma, it represents the the brightness in an image (the "black-and-white" or achromatic portion of the image). It has a lot of information about the image shape.
Cb and Cr are the blue-difference and red-difference chroma components.
This is the result of this conversion:
![alt text][image2]

####3. Extract features
The features extracted are the following:
* Get Histogram of Oriented Gradients (HOG)
* Get spatial features
* Get color histogram

#####a. Histogram of Oriented Gradients (HOG)
HOG permits that an image can be described by the distribution of intensity gradients or edge directions. It is some kind of signature of image shape.
I extracted the HOG features for the 3 channels of the YCbCr. Actually the Y channel (channel 0) has most of the data shape, however when I keept the 3 channels because I got a little bit better accurancy. 

The code for this step is contained in the fourth code cell (To call the `extract_features` function)  of the IPython notebook `"./Vehicle-detection-training.ipynb"`. The code about the for `skimage.hog()` is in the get_hog_features function of the `lesson_functions.py` file
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
* Channel 0:
![alt text][image3]

* Channel 1:
![alt text][image4]

* Channel 2:
![alt text][image5]

The parameters used for this are the following:
`orient = 9 #represents the number of orientation bins that the gradient information will be split up into in the histogram`
`pix_per_cell = 8 #the image grouped in cells of  8x8 pixels` 
`cell_per_block = 2 #specifies the local area over which the histogram counts in a given cell will be normalized`
`hog_channel = 'ALL' #It will process all the channels`

Note: Adding one more channel will increase a lot of the feature size, in this case i keep it only for a couple of accurancy, however it involves longer processing time .

#####b. Get spatial features
It consist in resize the image in order to get an smaller feature vector, and ap
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

######b. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

##### 4
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

