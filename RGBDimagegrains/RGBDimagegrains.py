import os, csv, time, argparse, sys
import cv2
import numpy as np
import tkinter as tk
import PCfunctions as func
from osgeo import gdal, osr, ogr
from scipy import ndimage as ndi
from skimage import color
from skimage import measure as meas
from skimage import segmentation as segm
from skimage import feature as feat
from skimage import morphology as morph
from skimage.morphology import (square, disk)
from skimage import filters as filt
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
# ignore some warnings thrown by sci-kit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy import ndimage as ndi

from imagegrains import grainsizing, plotting, segmentation_helper, data_loader, gsd_uncertainty
from cellpose import io
import pandas as pd
import matplotlib.pyplot as plt

resize_factor = 0.5

root = tk.Tk()
resize_factor = (1-resize_factor)+1
sys_w, sys_h = root.winfo_screenwidth()/resize_factor, root.winfo_screenheight()/resize_factor
root.destroy()
root.quit()
del root
print("System size width {:0.2f} and height {:0.2f} \n".format(sys_w, sys_h))


# depth image
width = 1280
height = 720
data_type = np.uint16
raw_file_path = "C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/aligned_depth_data.raw"
depth_data = np.fromfile(raw_file_path, dtype=data_type)
depth_img = depth_data.reshape((height, width))
# Scale depth values to 0-255, treating 0 as black
scaled_depth_img = ((depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) * 255).astype(np.uint8)

# Apply histogram equalization to enhance contrast and denose
def apply_contrast_stretching(image, low, high):
    # Ensure the low and high values are within the valid range [0, 255]
    # Create a copy of the image to avoid modifying the original
    stretched_image = image.copy()

    # Apply the contrast stretching to each pixel
    stretched_image = np.where(stretched_image < low, 0, stretched_image)
    stretched_image = np.where((low <= stretched_image) & (stretched_image <= high),
                              (255 / (high - low)) * (stretched_image - low), stretched_image)  # low < pixel depth value < high
    stretched_image = np.where(stretched_image > high, 255, stretched_image)
    
    return stretched_image
# Define the low and high values for contrast stretching
low_value = np.percentile(depth_img, 1)  # 1st percentile
high_value = np.percentile(depth_img, 99)  # 99th percentile
print(low_value,high_value)
# Apply contrast stretching to the image using the provided function
stretched_image = apply_contrast_stretching(depth_img, low_value, high_value) # low = low_value
stretched_image = np.uint16(stretched_image)
# Original Image and Histogram # y --> pixels x-->0-255, Distribution at each level
hist_original = cv2.calcHist([scaled_depth_img], [0], None, [256], [0, 256])
hist_stretched = cv2.calcHist([stretched_image], [0], None, [256], [0, 256])

# Display the original and stretched images side by side with their histograms
plt.figure(figsize=(12, 6))  # Adjust the figure size if needed

# Original Image and Histogram # y --> pixels x-->0-255, Distribution at each level
plt.bar(range(256), hist_original.flatten(), color='k', alpha=0.5, width=2, label='Original')
plt.bar(range(256), hist_stretched.flatten(), color='b', alpha=0.5, width=2, label='Stretched')
plt.title('Histogram Comparison')
plt.xlabel('Gray code')
plt.ylabel('Pixel number')
plt.gca().set_xlim([0, 255])
plt.gca().set_ylim([0, max(max(hist_original), max(hist_stretched))])
plt.legend()
plt.savefig('C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Histogram Comparison.png', dpi=300)

cv2.imwrite('C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Original.png', scaled_depth_img)
stretched_image_uint8 = stretched_image.astype(np.uint8)
cv2.imwrite('C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Stretched.png', stretched_image_uint8)

# RGB img
im = "C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/aligned_color_image.png"
img = cv2.imread(im)

RGB_w,RGB_h = img.shape[1],img.shape[0]
print("Original image data type:", img.dtype)
print("Image size width {:0.2f} and height {:0.2f} \n".format(RGB_w, RGB_h))



w, h = img.shape[1], img.shape[0]

scale_w = sys_w/w
scale_h = sys_h/h
dimensions = min(scale_h, scale_w)
window_w =int(w*dimensions)
window_h = int(h*dimensions)
print ("Resize width {:0.2f} and height {:0.2f}\n".format(window_w, window_h))
print("\nNon-local means filtering of color image")

start = time.time()

# RGB_denose (pebblecount default)
h_strength = 5
hForColor = 1 
imgdeNoise = cv2.fastNlMeansDenoisingColored(img, None, h_strength, hForColor, 7, 21)


# RGBwindow 
win_name = "Original Image ('q' for quit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 0, 0)
cv2.imshow(win_name, img)

win_name = "nlmean_denoise Image ('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 100, 0)
cv2.imshow(win_name, imgdeNoise)

gray = cv2.cvtColor(imgdeNoise, cv2.COLOR_BGR2GRAY)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/RGB_denose_gray.png", gray)
win_name = "nlmean_denoise_gray Image('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 200, 0)
cv2.imshow(win_name, gray)


otsu_th,_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Otsu threshold is {:0.0f}".format(otsu_th))


closeWIn = False
while True:
    while True:
        value = input("\nWhat percent of Otsu value({:0.0f})?(suggested 50):".format(otsu_th))
        try:
            value = int(value)
        except: 
            print("\nIncorrect input, should be an integer from 0-100\n")
        if isinstance(value, int):
           thresh = float(value)
           RGB_Otsu_threshold = otsu_th*(thresh/100)
           break
        else:
            print("\nIncorrect input, should be an integer from 0-100\n")

    
    gray_th =  gray > otsu_th*(thresh/100) #------------------------------------------------- equalized_depth_img
    gray_th_copy = np.copy(gray_th) # gray_th_copy is for edge detection, logical array
    gray_th = gray_th.astype(np.uint8) # gray_th is uint8, for mask 
    gray_th[gray_th == False] = 255
    gray_th[gray_th == True] = 0
    gray_th = np.dstack((gray_th, gray_th, gray_th))
    image_mask = cv2.addWeighted(gray_th, 0.8, imgdeNoise, 1, 0)
    win_name = "Otsu Shadow Mask Image ('y' keep, 'n' to try another, 'r' flash image)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while cv2.getWindowProperty(win_name, 0) >= 0 :
        cv2.imshow(win_name, image_mask)
        cv2.resizeWindow(win_name, window_w, window_h)
        cv2.moveWindow(win_name, 500, 0)
        k = cv2.waitKey(1)
        # only keep the threshold if the 'y' key is pressed
        if k == ord('y')& 0xFF :
            cv2.destroyWindow(win_name)
            closeWIn = True
            break
        elif k == ord('r')& 0xFF:
            timeout = time.time()+0.5
            while time.time() < timeout:
                cv2.namedWindow("Image Overlay", cv2.WINDOW_NORMAL)
                cv2.imshow("Image Overlay", imgdeNoise)
                print("Resize width {:0.2f} and height {:0.2f}".format(window_w, window_h))
                cv2.moveWindow("Image Overlay", 300, 0)
                cv2.waitKey(1)
            cv2.destroyWindow("Image Overlay")
        # ignore the threshold if 'n' or window is closed
        elif k == ord('n') & 0xFF:
            thresh = None
            cv2.destroyWindow(win_name)
            break
        elif cv2.getWindowProperty(win_name, 0) == -1:
            thresh = None
            break
    if closeWIn == True :
        break

# get end time 
end = time.time()
print("\nThat denoising_Otsu took about {:0.1f} seconds".format(end-start))

win_name = "Otsu Shadow Mask Image ('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 500, 0)
cv2.imshow(win_name, image_mask)

cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Otsu_mask.png", image_mask)

key = cv2.waitKey(0)

if key == ord('q') or key ==27:
    print("\nClose RGBimg window!!")

cv2.destroyAllWindows()



#　open img rgb mask   + depth gray window =============================================================================================
win_name = "Original Image ('q' for quit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 0, 0)
cv2.imshow(win_name, img)

win_name = "RGB Image mask  ('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 100, 0)
cv2.imshow(win_name, image_mask)

win_name = "denose depth Image('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 200, 0)
#cv2.imshow(win_name, depthdeNoise)
cv2.imshow(win_name, stretched_image_uint8)

start2 = time.time()

# depth denose
denose_equalized_depth_img = stretched_image[stretched_image != 0]  #----------------------------------------denose_equalized_depth_img

otsu_th_2,_ = cv2.threshold(denose_equalized_depth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Otsu threshold is {:0.0f}".format(otsu_th_2))


closeWIn = False
while True:
    while True:
        value = input("\nWhat percent of Otsu value({:0.0f})?(suggested 50):".format(otsu_th_2))
        try:
            value = int(value)
        except: 
            print("\nIncorrect input, should be an integer from 0-100\n")
        if isinstance(value, int):
           thresh = float(value)
           Depth_Otsu_threshold = otsu_th_2*(thresh/100)
           break
        else:
            print("\nIncorrect input, should be an integer from 0-100\n")
    gray_th_2 =  stretched_image < otsu_th_2*(thresh/100) #------------------------------------------------- equalized_depth_img
    # RGB otsu for depth senstive region 
    selected_region = gray[gray_th_2]
    otsu_th_selected, _ = cv2.threshold(selected_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    selected_region_thresh = selected_region > otsu_th_selected*(50/100)
    gray_th_2[gray_th_2] = selected_region_thresh  # change gray_th_2 shadow region true to false with otsu
    # otsu white mask 
    gray_th_copy_2 = np.copy(gray_th_2)
    gray_th_2 = gray_th_2.astype(np.uint8)
    gray_th_2[gray_th_2 == False] = 255
    gray_th_2[gray_th_2 == True] = 0
    gray_th_2 = np.dstack((gray_th_2, gray_th_2, gray_th_2))
    image_mask_2 = cv2.addWeighted(gray_th_2, 0.8, imgdeNoise, 1, 0)
    win_name = "Otsu depth2gray Mask Image ('y' keep, 'n' to try another, 'r' flash image)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while cv2.getWindowProperty(win_name, 0) >= 0 :
        cv2.imshow(win_name, image_mask_2)
        cv2.resizeWindow(win_name, window_w, window_h)
        cv2.moveWindow(win_name, 500, 0)
        k = cv2.waitKey(1)
        # only keep the threshold if the 'y' key is pressed
        if k == ord('y')& 0xFF :
            cv2.destroyWindow(win_name)
            closeWIn = True
            break
        elif k == ord('r')& 0xFF:
            timeout = time.time()+0.5
            while time.time() < timeout:
                cv2.namedWindow("Image Overlay", cv2.WINDOW_NORMAL)
                cv2.imshow("Image Overlay", stretched_image)
                print("Resize width {:0.2f} and height {:0.2f}".format(window_w, window_h))
                cv2.moveWindow("Image Overlay", 300, 0)
                cv2.waitKey(1)
            cv2.destroyWindow("Image Overlay")
        # ignore the threshold if 'n' or window is closed
        elif k == ord('n') & 0xFF:
            thresh = None
            cv2.destroyWindow(win_name)
            break
        elif cv2.getWindowProperty(win_name, 0) == -1:
            thresh = None
            break
    if closeWIn == True :
        break


# get end time 
end2 = time.time()
print("\nThat denoising_Otsu took about {:0.1f} seconds".format(end-start))

win_name = "Otsu Shadow Mask Image ('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 500, 0)
cv2.imshow(win_name, image_mask)

cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Otsu_mask_depth2gray.png", image_mask_2)


key = cv2.waitKey(0)

if key == ord('q') or key ==27:
    print("\nClose depth2grayimg window!!")

cv2.destroyAllWindows()

# combine mask ---which do you want chose?('a' : grain size for main object or 'b' : peeble for main object)
while True :
     combined_gray_chose = input("\n which do you want chose?('a' : imagegrains and RGBDgrains, 'b' : imagegrains)(a/b):")
     if combined_gray_chose == 'a' :
         #combined_gray_th = cv2.bitwise_and(gray_th, cv2.bitwise_not(gray_th_2))   # for grain size in depth mpa is background, in RGB map is object
         combined_gray_th = cv2.bitwise_and(gray_th, gray_th_2)                     # for sobel mask
         #combined_gray_copy = gray_th_copy | np.logical_not(gray_th_copy_2)
         combined_gray_copy = gray_th_copy | gray_th_copy_2                         # | == cv2.bitwise_or
         sobel3 = filt.sobel(gray)
         break
     elif combined_gray_chose == 'b' :
         break
     else :
         print("incorrect input, should be 'a' or 'b'")

combined_gray = cv2.cvtColor(combined_gray_th, cv2.COLOR_RGB2BGR)
combined_gray = np.mean(combined_gray, axis=-1).astype(np.uint8) # 3channel to 1 channel for morph thin 
combined_mask = cv2.addWeighted(combined_gray_th, 0.8, imgdeNoise, 1, 0)


win_name = "Combined Mask Image ('q' for exit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, window_w, window_h)
cv2.moveWindow(win_name, 500, 0)
cv2.imshow(win_name, combined_mask)

cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/combined_mask.png", combined_mask)

# wait for key 
key = cv2.waitKey(0)

# quit
if key == ord('q') or key == 27:
    print("\nClose combined mask window!!")


cv2.destroyAllWindows()


### RGB　sobel edges(only keep the ROI edge detection, like gray_th_copy > threshold)
# tophat edges
print("RGB Black tophat edge detection")
tophat_th = 90 
tophat = morph.black_tophat(gray, footprint=disk(1))
tophat = tophat < np.percentile(tophat, tophat_th)
tophat = morph.remove_small_holes(tophat, area_threshold=5, connectivity=2)
if not np.sum(tophat) == 0:
    foo = func.featAND_fast(gray_th_copy, tophat)
    gray_th_copy = np.logical_and(foo, gray_th_copy)
# canny edges
print("RGB Canny edge detection")
canny_sig = 2
canny = feat.canny(gray, sigma=canny_sig)
canny = np.invert(canny)
foo = func.featAND_fast(gray_th_copy, canny)
gray_th_copy = np.logical_and(foo, gray_th_copy)

print("RGB Sobel edge detection")
sobel_th = 90
sobel = filt.sobel(gray)
sobel = sobel < np.percentile(sobel, sobel_th)
sobel = morph.remove_small_holes(sobel, area_threshold=5, connectivity=2)
sobel = morph.thin(np.invert(sobel))
sobel = np.invert(sobel)  # edge -->True, not edge -->False   
foo = func.featAND_fast(gray_th_copy, sobel) # combine the True reigon(intersection) 
gray_th_copy = np.logical_and(foo, gray_th_copy)


rgb_edges = (gray_th_copy / np.max(gray_th_copy) * 255).astype(np.uint8)
cv2.imshow("RGB sobel edge", rgb_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/RGB_sobel_edge.png", rgb_edges)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/RGB_edge.png", rgb_edges)


### Depth sobel edges(only keep the ROI edge detection, like gray_th_copy > threshold)
print("Depth Sobel edge detection")
sobel_th2 = 20
sobel2 = filt.sobel(stretched_image) # sobel is detecting by gray gradient
sobel2 = sobel2 < np.percentile(sobel2, sobel_th2) # logical array
sobel2 = morph.remove_small_holes(sobel2, area_threshold=5, connectivity=2)
sobel2 = np.invert(sobel2)  # edge -->True, not edge -->False    
foo2 = func.featAND_fast(gray_th_copy_2, sobel2) # combine the True reigon(intersection) 
gray_th_copy_2 = np.logical_and(foo2, gray_th_copy_2) # updat the gray_th for next move

depth_sobel_edges = (gray_th_copy_2 / np.max(gray_th_copy_2) * 255).astype(np.uint8) # True * 255 
cv2.imshow("Depth sobel edge", depth_sobel_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Depth_sobel_edge.png", depth_sobel_edges)


### Conbined sobel edges(only keep the ROI edge detection, like gray_th_copy > threshold)
print("Sobel combined edge detection")
sobel_th3 = 90
#sobel3 = filt.sobel(gray)
sobel3 = sobel3 < np.percentile(sobel3, sobel_th3)  # sobel3 is logical array, threshold 100 --> 0 : edge intensity storng --> weak
sobel3 = morph.remove_small_holes(sobel3, area_threshold=5, connectivity=2)
sobel3 = morph.thin(np.invert(sobel3)) # image need a 2-dimension
sobel3 = np.invert(sobel3)  # edge -->True, not edge -->False    
foo3 = func.featAND_fast(combined_gray_copy, sobel3) # combine the True reigon(intersection) 
combined_gray_copy = np.logical_and(foo3, combined_gray_copy) # update the gray_th for next move

combined_edges = (combined_gray_copy / np.max(combined_gray_copy) * 255).astype(np.uint8)
cv2.imshow("Combined sobel edge", combined_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Combined_sobel_edge.png", combined_edges)



# RGB Sobel cleanup the mask
min_size = 10
RGB_master_mask = morph.remove_small_objects(gray_th_copy, min_size=min_size, connectivity=2)
RGB_master_mask  = morph.erosion(RGB_master_mask, footprint=square(3))
RGB_master_mask  = morph.dilation(RGB_master_mask, footprint=square(2))
RGB_master_mask  = segm.clear_border(RGB_master_mask) # this function deleate too much reigon 
RGB_master_mask  = morph.remove_small_objects(RGB_master_mask, min_size=min_size, connectivity=2)
# make sure we didn't accidently add any definite edges back in
RGB_master_mask [gray_th_copy == False] = False
RGB_master_mask= (RGB_master_mask/ np.max(RGB_master_mask) * 255).astype(np.uint8)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/RGB_master_mask.png", RGB_master_mask)

# Depth Sobel cleanup the mask 
min_size = 10
Depth_master_mask = morph.remove_small_objects(gray_th_copy_2, min_size=min_size, connectivity=2)
Depth_master_mask = morph.erosion(Depth_master_mask , footprint=square(3))
Depth_master_mask = morph.dilation(Depth_master_mask , footprint=square(2))
Depth_master_mask = segm.clear_border(Depth_master_mask) # this function deleate too much reigon 
Depth_master_mask = morph.remove_small_objects(Depth_master_mask , min_size=min_size, connectivity=2)
# make sure we didn't accidently add any definite edges back in
Depth_master_mask  [gray_th_copy_2 == False] = False
Depth_master_mask = (Depth_master_mask/ np.max(Depth_master_mask) * 255).astype(np.uint8)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Depth_master_mask.png", Depth_master_mask)



#  ===================================================================================================
#  test 2

# sobel combined 
combined_sobel_gray_copy = np.logical_and(gray_th_copy, gray_th_copy_2)
combined_edges_2 = (combined_sobel_gray_copy / np.max(combined_sobel_gray_copy) * 255).astype(np.uint8)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Combined_sobel_edge_2.png", combined_edges_2)


# cleanup the mask
#min_size = 10
master_mask_2 = morph.remove_small_objects(combined_sobel_gray_copy, min_size=min_size, connectivity=2)
master_mask_2 = morph.erosion(master_mask_2, footprint=square(3))
master_mask_2 = morph.dilation(master_mask_2, footprint=square(4))
master_mask_2 = segm.clear_border(master_mask_2) # this function deleate too much reigon 
master_mask_2 = morph.remove_small_objects(master_mask_2, min_size=min_size, connectivity=2)
# make sure we didn't accidently add any definite edges back in
master_mask_2[combined_gray_copy == False] = False
master_mask_2 = (master_mask_2 / np.max(master_mask_2) * 255).astype(np.uint8)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/Master_mask_2.png", master_mask_2)

### Grainsize 

# eliminate any regions that are smaller than cutoff value
cutoff = 10 
tmp, num = ndi.label(master_mask_2)
for region in meas.regionprops(tmp):
    grain_dil_ = morph.dilation(region.image, footprint=square(2)).astype(int)
    grain_dil_ = np.pad(grain_dil_, ((1, 1), (1,1)), 'constant')
    b_ = meas.regionprops(grain_dil_)[0].minor_axis_length
    a_ = meas.regionprops(grain_dil_)[0].major_axis_length
    if b_ < float(cutoff) or a_ < float(cutoff):
        idxs = region.coords
        idxs = [tuple(i) for i in idxs]
        for idx in idxs:
            tmp[idx] = 0
idx = np.where(tmp == 0)
master_mask_2[idx] = 0

# get all the grains in the final mask
grains = []
polys = []
coordList = []
print("Getting grain properties")
labels, _ = ndi.label(master_mask_2)
for grain in meas.regionprops(labels):
    # dilate the grain before getting measurements
    grain_dil = morph.dilation(grain.image, footprint=square(2)).astype(int)
    grain_dil = np.pad(grain_dil, ((1, 1), (1,1)), 'constant')
    b = meas.regionprops(grain_dil)[0].minor_axis_length
    a = meas.regionprops(grain_dil)[0].major_axis_length
    # get ellipse ring coordinates
    y0, x0 = grain.centroid[0], grain.centroid[1]
    orientation = grain.orientation - np.pi/2
    phi = np.linspace(0,2*np.pi,50)
    X = x0 + a/2 * np.cos(phi) * np.cos(-orientation) - b/2 * np.sin(phi) * np.sin(-orientation)
    Y = y0 + a/2 * np.cos(phi) * np.sin(-orientation) + b/2 * np.sin(phi) * np.cos(-orientation)
    # convert coordinates
    tupVerts = list(zip(X, Y))
    p = Path(tupVerts)
    # append ellipse as shapely polygon
    x, y = zip(*p.vertices)
    poly = Polygon([(i[0], i[1]) for i in list(zip(x, y))])
    polys.append(poly)
    # append list of grain coordinates
    #  for later removal if misfit/overlap
    grain_coords = [(i[0], i[1]) for i in grain.coords]
    coordList.append(grain_coords)
    # also get the percent difference in area (misfit)
    if poly.area != 0:
        perc_diff_area = ((poly.area-grain.filled_area)/poly.area)*100
    else:
    # Handle the case when poly.area is zero
    # You can choose to assign a default value or take other actions
        perc_diff_area = 0  # For example, assigning zero as default value
    # append the grain
    grains.append((y0, x0, b, a, orientation, grain.filled_area, poly.area, perc_diff_area))

# remove grains based on centroid inside of another grain and percentage overlap
# TODO: THIS IS SLOW
perc_overlap = 15
remove_indices = []
print("Removing overlapping grains")
# Removing overlapping grains
for index, poly in enumerate(polys):
    x, y = poly.centroid.coords.xy[0][0], poly.centroid.coords.xy[1][0]
    check_pt = Point(x, y)
    for index_check, poly_check in enumerate(polys):
        # Check the ellipse against all other ellipses except itself
        if not index == index_check:
            # Check if the centroid is contained in another poly
            if poly_check.contains(check_pt):
                remove_indices.append(index)
            else:
                # Check if the overlap with another poly is greater than the threshold
                if poly.area != 0 and poly.intersection(poly_check).area / poly.area > perc_overlap / 100:
                    remove_indices.append(index)

# also find indices where the percent misfit is above some threshold and remove these
misfit_threshold = 30
print("Removing misfit grains")
for index, grain in enumerate(grains):
    if index in remove_indices:
        continue
    elif np.abs(grain[7]) >= misfit_threshold:
        remove_indices.append(index)


# use the indices to remove the offending grains from the list and from the mask
label_fixed = labels.copy().astype(bool)
grains = [i for j, i in enumerate(grains) if j not in remove_indices]
for index in remove_indices:
    for i in coordList[index]:
        label_fixed[i] = False


# output LABELS
print("Output LABELS")
# save out as raster or image
labels, _ = ndi.label(label_fixed)
img_labels = (color.label2rgb(labels, bg_label=0, bg_color=[1, 1, 1])*255).astype(np.uint8)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/final_label.png", img_labels)


# get end time
end = time.time()
print("\nThat took about {:.0f} minutes, you counted {:d} pebbles!\n".format(end/60-start/60, len(grains)))

# RGBimage(RGBDgrains result)
black_y,black_x = np.where(label_fixed)
img_with_RGBDgrains = np.copy(img)
img_with_RGBDgrains[black_y, black_x]=(0, 0, 0)
cv2.imwrite("C:/imagegrains_RGBDgrains/RGBDgrains/0603-5/RGBDgrains_result.jpg", img_with_RGBDgrains)

#=========================================imagegrains and RGBDgrains============================================
### Load image
# Path to the images # path data input commnd is different to pebblecounts, just need a file path
image_path = "C:/imagegrains_RGBDgrains/imagegrains/0603-5"
# Path to the segementation model
model_path = "C:/Users/a0952/imagegrains/models/full_set_1.170223"

### Segmentation
# Run segmentation 
segmentation_helper.predict_dataset(image_path, model_path, mute=True, return_results=False, save_masks=True)
# Get image and segementation mask paths...
imgs,_,preds = data_loader.dataset_loader(image_path, pred_str="pred")
# ... and plot them (plotting this library only for jupyter)
plt.figure(figsize=(7, 7))
# pred[0]-->label mask
plotting.plot_single_img_pred(imgs[0], preds[0], show=True)
#print(type(preds[0]))---> <class 'str'>
#print(len(preds[0]))---> <80> file name['C:/imagegrains_RGBDgrains/imagegrains/0603-5\\RGBDgrains_result_full_set_1_pred.tif'], not pixels 
plt.savefig(image_path+'/pred.png',dpi=300)


# imagegrains mask(tif)
imagegrains = io.imread("C:/imagegrains_RGBDgrains/imagegrains/0603-5/aligned_color_image_full_set_1_pred.tif")
print(imagegrains)
print(type(imagegrains))
count_imagegrains=np.max(imagegrains)
print(count_imagegrains)
# calcute the imagegrains mask pixels
imagegrains_mask_pixels = []
imagegrains_mask = 0
for imagegrains_mask in range(count_imagegrains+1):
    imagegrains_label = imagegrains == imagegrains_mask
    imagegrains_mask_pixels_count = np.sum(imagegrains_label)
    imagegrains_mask_pixels.append((imagegrains_mask, imagegrains_mask_pixels_count ))
    imagegrains_mask +=1

#print(imagegrains_mask_pixels)
rows, cols = imagegrains.shape
print(rows, cols)

# calcute the RGBDgrains mask pixel
RGBDgrains_mask_pixel =[]
for label in range (1, np.max(labels)+1):
    RGBD_mask = labels == label
    RGBD_mask_pixel_count = np.sum(RGBD_mask)
    center_y, center_x = ndimage.measurements.center_of_mass(RGBD_mask)
    RGBDgrains_mask_pixel.append((label, RGBD_mask_pixel_count, (center_y, center_x)))
    print("RGBD_mask_pixel_count",RGBDgrains_mask_pixel)

RGBDgrains_mask_pixel_array = np.array(RGBDgrains_mask_pixel, dtype=object)

# imagegrains and RGBDgrains pixel overlap remove
for label in range(1, np.max(labels)+1):
    # RGBDgrains
    RGBD_mask = labels == label
    indices = np.where(RGBDgrains_mask_pixel_array[:, 0] == label)[0]
    pixel_number = RGBDgrains_mask_pixel_array[indices[0], 1]
    pixel_center = RGBDgrains_mask_pixel_array[indices[0], 2]
    center_y1 = pixel_center[0]
    center_x1 = pixel_center[1]
    print('label', label)
    print('pixel_center:', pixel_center)
    print('center_y', center_y1)
    print('center_x', center_x1)
    # imagegrains 
    imagegrains_mask_pixels_array = np.array(imagegrains_mask_pixels)
    indices1 = np.where(imagegrains_mask_pixels_array[:, 0] == imagegrains[int(center_y1), int(center_x1)])[0]
    imagegrains_mask_pixel_value = imagegrains_mask_pixels_array[indices1[0], 1]
    count_imagegrains = count_imagegrains + 1
    if imagegrains[int(center_y1), int(center_x1)] != 0:
        if pixel_number >  imagegrains_mask_pixel_value:
            print(f"Pixel number ({pixel_number}) is greater than imagegrains mask pixel value ({imagegrains_mask_pixel_value})")
            for y, x in np.transpose(np.where(RGBD_mask)):
                mask_indices = imagegrains == imagegrains[y, x]
                original_mask_number = imagegrains[y, x]
                imagegrains[mask_indices] = 0
                imagegrains[y, x] = count_imagegrains
    else :
        for y, x in np.transpose(np.where(RGBD_mask)):
            mask_indices = imagegrains == imagegrains[y, x]
            original_mask_number = imagegrains[y, x]
            imagegrains[mask_indices] = 0
            imagegrains[y, x] = count_imagegrains


io.imsave("C:/imagegrains_RGBDgrains/imagegrains/0603-5/update_img/pred_modified_test_array.tif", imagegrains)
print(np.max(imagegrains))
print(imagegrains.shape)
print(type(imagegrains))
# update the pred mask
### combined imagegrains and RGBD
# file name need 'pred'
update_preds="C:/imagegrains_RGBDgrains/imagegrains/0603-5/update_img/pred_modified_test_array.tif"
update_img = "C:/imagegrains_RGBDgrains/imagegrains/0603-5/update_img/aligned_color_image.jpg"
update_img_path = "C:/imagegrains_RGBDgrains/imagegrains/0603-5/update_img"
plt.figure(figsize=(7, 7))
plotting.plot_single_img_pred(update_img, update_preds, show=True)
plt.savefig(update_img_path+'/pred_with_RGBDgrains.png',dpi=300)



### Calculate grain properties
#measure the grains and save the results
imagegrains_grains,grains_props,grain_id = grainsizing.batch_grainsize(update_img_path, mask_str='pred_modified', fit_method='convex_hull', mute=True,return_results=True)
print("Number of elements in grains_props:", len(grains_props))
#set the resolution in mm/px
resolution = 1.0744

#scale with the image resolution
grainsizing.re_scale_dataset(update_img_path, resolution=resolution, gsd_str='_grains', save_gsds=True)
#define plot elements
elements= ['image','mask','ellipse_b','ellipse_a']
#plot
plt.figure(figsize=(7,7))
plotting.all_grains_plot(io.imread(update_preds),props=grains_props[0],elements=elements,image=io.imread(update_img),title=grain_id[0])
plt.savefig(update_img_path+'/resampled_grains.png',dpi=300)

### Grain size distribution and uncertainty
#load the resampled grains
grains_scaled = data_loader.load_grain_set(update_img_path, gsd_str='pred_modified_test_array_grains_re_scaled')
#calculate the gsd for one column (e.g. b-axis with ellispe fit)
column_name = 'ell: b-axis (mm)'
#load results(csv) and get gsd for each image for a specific axis column
grains_scaled_gsd, grains_scaled_id = grainsizing.gsd_for_set(grains_scaled, column=column_name)
plt.figure(figsize=(10, 2.5))
for i in range(len(grains_scaled_gsd)):
    plt.subplot(1,len(grains_scaled_gsd),i+1)
    plotting.plot_gsd(grains_scaled_gsd[i],color='b',length_max=80,orientation='vertical',label_axes=True,units= 'mm',
                      title=grains_scaled_id[i])
plt.savefig(update_img_path+'/GSD.png',dpi=300)

#set uncertainty method
method = 'bootstrapping'
#calculate the gsd and uncertainty
res_dict_bs = gsd_uncertainty.dataset_uncertainty(gsds=grains_scaled,num_it=1000,method=method,mute=True,column_name=column_name,save_results=False,return_results=True,sep=',',gsd_id=grain_id)

#compile full results into a dataframe and save it
full_gsd_results = pd.DataFrame({f'{column_name}_perc_lower_CI':res_dict_bs[grain_id[0]][2],
                                f'{column_name}_perc_median':res_dict_bs[grain_id[0]][0],
                                f'{column_name}_perc_upper_CI':res_dict_bs[grain_id[0]][1],
                                f'{column_name}_perc_value':res_dict_bs[grain_id[0]][3]})
#create output directory
out_dir = update_img_path + '/GSD_uncertainty/'
os.makedirs(out_dir,exist_ok=True)
#save the results
full_gsd_results.to_csv(f'{out_dir}/{grain_id[0]}_{method}_full_uncertainty.csv')

# calculate and save key percentile summary (default=b-axis)
summary_df = grainsizing.summary_statistics(grains_scaled,grain_id,res_dict=res_dict_bs,data_id=grain_id[0],save_summary=True)
summary_df.head()
#plot gsd 
plotting.plot_gsd(res_dict_bs[grain_id[0]][3],color='b',length_max=80,orientation='vertical',label_axes=True,units= 'mm')
#plot uncertainty
plotting.plot_gsd_uncert(res_dict_bs[grain_id[0]],color='b')
plt.legend([column_name,f'95% CI from {method}'])
plt.title(grain_id[0])