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
* `orient = 9 #represents the number of orientation bins that the gradient information will be split up into in the histogram` 
* `pix_per_cell = 8 #the image grouped in cells of  8x8 pixels` 
* `cell_per_block = 2 #specifies the local area over which the histogram counts in a given cell will be normalized`
* `hog_channel = 'ALL' #It will process all the channels`

Note: Adding one more channel will increase a lot of the feature size, in this case i keep it only for a couple of accurancy, however it involves longer processing time.

#####b. Get spatial features
It consist in resize the image in order to get an smaller feature vector (in 1D version of the same image). I resized the image to 32 x 32 pixeles with 3 channels. the return will be 32 x 32 x3 = 3072. The code about the this feature extraction is  in `bin_spatial` function of the `lesson_functions.py` file.

#####c.Get color histogram
Compute the histogram of the color channels separately and then concatenate them, The number of bins that I used is 16, the result will be 16 x 3 = 48 values.Here is an example of two of each of the `vehicle` and `non-vehicle` classes:

#####d. Concatenate all the features
I concatenated the 3 extracted feature in order to get a 1 dimension array and then normalize the result.
This with the following line `X_scaler = StandardScaler().fit(X)` getting the scaler with `X_scaler = StandardScaler().fit(X)`

####4. Training the data
I performed the following steps (The code for this step is contained in the fourth code cell of the IPython notebook `"./Vehicle-detection-training.ipynb"`.):
* Define the labels vector (car and notcars)
* Split up data into randomized training and test sets, this in order to validate the training classifier and to know if this can be generalized. The training set is 20 % of all the data: `Train set: 14205 test set: 3552`
* for the classification I used [sklearn.svm.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) .

####5. Save the parametes and trained classifier in a pickle file
The data saved in the parameters.p file is the following:

*spatial_feat # Spatial features on or off
*hist_feat # Histogram features on or off
*hog_feat # HOG features on or off
*orient    #for HOG
*pix_per_cell  #for HOG
*cell_per_block #for HOG 
*hog_channel # for HOG
*spatial_size #for spatial_features
*hist_bins #for color_hist
*X_scaler #Scaler for final feature vector
*svc # trained classifier
*color_space 

## Implementation
All the code of this section is in IPython notebook `"./Vehicle-detection-implementation.ipynb"`
####1. Apply a distortion correction to raw images

####2. Sliding Window for search
I performed Sliding for search veehicles in an image. The code about the this feature extraction is  in `slide_window` function of the `lesson_functions.py` file.
This is how I performed the Sliding Window:

Electing the right Sliding Window will affect drastically the time of processing, because it will mean the times that we need to check if there is car or not (perform all the process). For all I elected an overlap of 0.75.

####3. Extract features
It is the same that the Extract features explained before, however all this is in the `find_cars` funtion of the IPython notebook `"./Vehicle-detection-implementation.ipynb"`.). This because it applies some operations onces and not for each slided window.
Note: in this case I changed the scale of the image, for the training set I used PNG files that are float (0..1) and the video reads JPEG files that are int values (0..255) `img = img.astype(np.float32)/255`

####4. Find car (Process performed on each Window)
#####a.Apply feature extraction,  svc classification to get boxes where posible there is a car
#####b.filter for false positives
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded (in this case it is 2) that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. I also added in the `draw_labeled_bboxes` function some validations if two final boxes are very close, it can join them.
The video result shows the heatmap from a series of frames of video.

####2. In order to optimize the accurancy I added to recorded the positions of positive detections bounding boxes from the previous frame.
### Here are 2 frames and their corresponding heatmaps:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

### Here the resulting bounding boxes are drawn onto the last frame in the series:

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

