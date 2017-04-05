##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./img/cars.png
[image2]: ./img/notcars.png
[image3]: ./img/cars_normal.png
[image4]: ./img/notcars_normal.png
[image5]: ./img/cars_hog_ch1.png
[image6]: ./img/cars_hog_ch2.png
[image7]: ./img/cars_hog_ch3.png
[image8]: ./img/notcars_hog_ch1.png
[image9]: ./img/notcars_hog_ch2.png
[image10]: ./img/notcars_hog_ch3.png
[image11]: ./img/boxes.png
[image12]: ./img/cars_heat1.png
[image13]: ./img/cars_heat2.png
[image14]: ./img/cars_heat3.png
[image15]: ./img/cars_heat4.png
[image16]: ./img/cars_heat5.png
[image17]: ./img/cars_heat6.png
[image18]: ./img/window.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! You can access it also [here](https://github.com/JosuVicente/CarND-Vehicle-Detection/blob/master/writeup_template.md)

Note that I use the following files on the project:
* P5_final.ipynb: has the project flow from loading data, training the model to applying it to the project video and displaying it.
* P5_functions.py: has the functions that support the main notebook on vehicle detection.
* P4_functions.py: has the functions from previous Lane Finding project.
* P5_visualization.ipynb: is a notebook where images from differente steps of this project flow are being displayed.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to extract the HOG features is contained in the `P5_functions.py` file in the function `get_hog_features()` which is called from the function `extract_features()` from same file.

I started by reading in all the `vehicle` and `non-vehicle` images.  The code for this is contained on second cell of `P5_final.ipynb` Here is information from where the data is loaded from, the amount of examples of each type and some images of each of the `vehicle` and `non-vehicle` classes:
```
Car Images > img/vehicles/GTI_Far/*
Car Images > img/vehicles/GTI_Left/*
Car Images > img/vehicles/GTI_MiddleClose/*
Car Images > img/vehicles/GTI_Right/*
Car Images > img/vehicles/KITTI_extracted/*
Not Car Images > img/non-vehicles/Extras/*
Not Car Images > img/non-vehicles/GTI/*
Total car images: 8792
Total not car images: 8968
```

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here there are some examples using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for each channel and for cars and not cars:


![alt text][image3]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image4]
![alt text][image8]
![alt text][image9]
![alt text][image10]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I finally chose these:
```
orientations=10
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code to train the classifer is located on third cell of `P5_final.ipynb` although it´s being called from fifth cell.
Parameters used are configured on fourth cell and are like follows:
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
x_start_stop = [None, None] # Min and max in x to search in slide_window()
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
```
Features are extracted using function `extract_features()` located on `P5_functions.py`. I ended using a combination of spatial, histogram and hog features.

I trained a linear SVM using the features extracted and I obtained this accuracy on the test set:
```
Test accuracy:  0.9893
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search windows of 64px size between 400 and 656 y position with an overlap of 0.5 like on image below:

![alt text][image18]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image11]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://raw.githubusercontent.com/JosuVicente/CarND-Vehicle-Detection/master/project_video_out_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in last 8 frames of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. This is done on cell 6 and 7 of `P5_final.ipynb`

For the final video I make use of functions from previous project and merge the lanes detected with the vehicles to output the final video.

### Here are six frames with initial boxes, their corresponding heatmaps and the resulting bounding boxes:
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the implementation works pretty well although it´s only trained to detect cars of certain characteristics and not other road vehicles. One thing I´d like to improve if the amount of time that takes to process each frame. On my PC processing the 50 seconds project video including lane detection takes about 8 minutes so I definitely would like to improve that.
