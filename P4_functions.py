#importing some useful packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

### Calibration

def calibrate():    
    cor_h = 9
    cor_v = 6
    objp = np.zeros((cor_v*cor_h,3), np.float32)
    objp[:,:2] = np.mgrid[0:cor_h, 0:cor_v].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cor_h,cor_v), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cor_h,cor_v), corners, ret)
        
    return objpoints, imgpoints

### Functions

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max   
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(imgimg, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(imgimg, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobelxy = np.sqrt(sobel_x**2 + sobel_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan_sobel = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    scaled_sobel = arctan_sobel#np.uint8(255*arctan_sobel/np.max(arctan_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def hls_threshold(img, channel, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    if (channel == 'H'):
        select = hls[:,:,0]
    elif (channel == 'L'):
        select = hls[:,:,1]
    else:
        select = hls[:,:,2]
    
    binary_output = np.zeros_like(S)
    binary_output[(select > thresh[0]) & (select <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def thresold_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binaries = []
    
    color_binaries.append(color_binary)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    color_binaries.append(combined_binary)
    return color_binaries


### Perspective Transform

def unwarp(img):    
    img_size = (img.shape[1], img.shape[0])
    
    width, height = img_size
    offset = 250
    attempt_sel = 2
    
    if (attempt_sel == 1):
        ## Atempt 1
        src = np.float32([
            [  588,   446 ],
            [  691,   446 ],
            [ 1126,   673 ],
            [  153 ,   673 ]])
    elif (attempt_sel == 2):
        ## Attempt 2
        src = np.float32([
            [  598,   448 ],
            [  685,   448 ],
            [ 1055,   677 ],
            [  252 ,   677 ]])
    elif (attempt_sel == 3):
        ## Attempt 3
        sideMargin = 0.14 # percentage of the sides to skip
        topMargin = 0.62 # percentage of the top to skip
        midMargin =0.039 # percentage of mid horizon to include    
        bottomMargin = 0
        shift_x = 20
        src = np.float32([(int((img_size[0]/2)-(img_size[0]*midMargin) + shift_x), int(img_size[1]*topMargin)), 
                           (int((img_size[0]/2)+(img_size[0]*midMargin) + shift_x), int(img_size[1]*topMargin)), 
                           (img_size[0]-int(img_size[0]*sideMargin) + shift_x,img_size[1]-int(img_size[1]*bottomMargin)),
                           (int(img_size[0]*sideMargin) + shift_x,img_size[1]-int(img_size[1]*bottomMargin))])
    elif (attempt_sel == 4):
        ##Attempt 4
        sideMargin = 0.12 # percentage of the sides to skip
        topMargin = 0.62 # percentage of the top to skip
        midMargin =0.04 # percentage of mid horizon to include    
        bottomMargin = 0.06
        shift_x = 0
        src = np.float32([(int((img_size[0]/2)-(img_size[0]*midMargin) + shift_x), int(img_size[1]*topMargin)), 
                           (int((img_size[0]/2)+(img_size[0]*midMargin) + shift_x), int(img_size[1]*topMargin)), 
                           (img_size[0]-int(img_size[0]*sideMargin) + shift_x,img_size[1]-int(img_size[1]*bottomMargin)),
                           (int(img_size[0]*sideMargin) + shift_x,img_size[1]-int(img_size[1]*bottomMargin))])
    
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #print(M)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # Return the resulting image and matrix
    return warped, M, Minv, src, dst

### Preprocessing pipeline

def preprocess(img, cor_h, cor_v, mtx, dist):
    img = cv2.imread(fname)
    pipe = thresold_pipeline(img)
    warped, perspective_M, perspective_Minv, src, dst = unwarp(pipe[1])
    return pipe, warped, perspective_M, perspective_Minv, img, src, dst
    
### Lane finding - Sliding Windows and Fit Polynomial

def findlines(prep, left_fit, right_fit):

    binary_warped = prep[2]
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]*(2/3))
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Choose the number of sliding windows
    if (len(left_fit) == 0 or len(right_fit) == 0):
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)    

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)    
    else:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_curverad_p, right_curverad_p, left_curverad, right_curverad, offset = get_curvature_and_offset(ploty, left_fitx, right_fitx, left_fit, right_fit, prep[0].shape[1])
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    return ploty, left_curverad, right_curverad, offset, pts_left, pts_right
   
def get_curvature_and_offset(ploty, left_fitx, right_fitx, left_fit, right_fit, x_shape):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad_p = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad_p = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    
    # Now our radius of curvature is in meters
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #Calculate offset
    image_center = x_shape  /2
    off_left = image_center - left_fit[0]
    off_right = right_fit[0] - image_center    
    offset_pix = (off_right - off_left)/2
    offset = xm_per_pix*offset_pix   
    return left_curverad_p, right_curverad_p, left_curverad, right_curverad, offset
    

def get_offset(leftx, rightx, width):
   
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #Calculate offset
    image_center = width  /2
    off_left = image_center - leftx
    off_right = rightx - image_center    
    offset_pix = (off_right - off_left)/2
    offset = xm_per_pix*offset_pix    
  
    return offset

def locate_lane(img, t_last_data, mtx, dist):
    left_fit = []
    right_fit = []

    img_size = (img.shape[1], img.shape[0])

    img_und = cv2.undistort(img, mtx, dist, None, mtx)
    img_thres = thresold_pipeline(img_und)
    warped, perspective_M, perspective_Minv, src, dst = unwarp(img_thres[1])
    prep = [img, img_thres, warped, perspective_M, perspective_Minv, src, dst]
    ploty, left_curverad, right_curverad, offset, pts_left, pts_right = findlines(prep, left_fit, right_fit)

    if(len(t_last_data)==0):
        t_last_data = [pts_left, pts_right,left_curverad,right_curverad]

    min_curv = 100.
    max_curv = 10000.
    if (left_curverad<min_curv or left_curverad>max_curv):
        #print('left wrong')
        pts_left = t_last_data[0]
        left_curverad = t_last_data[2]

    if (right_curverad<min_curv or right_curverad>max_curv):
        #print('right wrong')
        pts_right = t_last_data[1]
        right_curverad = t_last_data[3]
    pts = np.hstack((pts_left, pts_right))


    t_last_data = [pts_left, pts_right, left_curverad, right_curverad]


    offset_v2 = get_offset(pts_left[0][0][0], pts_right[0][0][0], prep[0].shape[1])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    video_image = img
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspective_Minv, (video_image.shape[1], video_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(video_image, 1, newwarp, 0.3, 0)

    #cv2.putText(result,'LC: {0:.2f} m'.format(left_curverad), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
    #cv2.putText(result,'RC: {0:.2f} m'.format(right_curverad), (30,70), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
    #cv2.putText(result,'Offset: {0:.2f} m'.format(offset_v2), (30,110), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

    #result[10:210,1070:1270,:] = cv2.resize(color_warp,(200,200))

    global last_data 
    last_data = t_last_data
    return result
