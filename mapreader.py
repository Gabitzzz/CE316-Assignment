#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.

import sys
import cv2
import numpy as np
import math

### Required Constant HSV values for the color thresholding

BLUE_BG_HSV_VALUES = [51, 108, 23, 255, 0, 252]  ## H_min, H_max, S_min, S_max, V_min, V_max
RED_POINTER_HSV_VALUES = [153, 179, 37, 255, 188, 255]
RED_POINTER_HSV_VALUES = [176, 179, 70, 255, 0, 255]



def getHSV_mask(img, HSV_values):
    '''Function To return mask in betweeen HSV values range'''

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([HSV_values[0], HSV_values[2], HSV_values[4]])
    upper_hsv = np.array([HSV_values[1], HSV_values[3], HSV_values[5]])
    mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)

    return mask

def distance_between_points(pt1,pt2):
    '''Function that returns the distance between two coordinates'''
    x1,y1 = pt1
    x2,y2 = pt2
    dist = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
    return dist

def extract_corners_list(maxCnt):
    '''Function to extract corners coordinates from contour'''
    epsilon = 0.02*cv2.arcLength(maxCnt,True)
    approx = cv2.approxPolyDP(maxCnt,epsilon,True)    
    approx = approx.tolist()
    
    corner_list = []
    for pnt in approx:
        pnt = pnt[0]
        corner_list.append(pnt)
        # cv2.circle(inputImg, tuple(pnt),7,(0,0,255),-1)

    return corner_list

def reOrient_map(map_mask, inputImg):
    '''Function that takes argument mask extracted for map and reorient the map'''
    
    contours, heirarchy = cv2.findContours(map_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours)>0:
        # Get maximum area contour which represents map only.
        maxCnt = max(contours, key = cv2.contourArea)        
        epsilon = 0.01*cv2.arcLength(maxCnt,True)
        approx = cv2.approxPolyDP(maxCnt,epsilon,True)    
        approx = approx.tolist()    
        corner_list = []
        for pnt in approx:
            pnt = pnt[0]
            corner_list.append(pnt)
            cv2.circle(inputImg, tuple(pnt),7,(0,0,255),-1)

        if len(corner_list)==4:
            dist_list = []
            for i in range (len(corner_list)-1):
                pt1 =  corner_list[i]
                pt2 = corner_list[i+1]
                dist = distance_between_points(pt1, pt2)
                dist_list.append(dist)
            dist_last = distance_between_points(corner_list[0], corner_list[-1])
            dist_list.append(dist_last)
            map_h = int(min(dist_list))
            map_w = int(max(dist_list))
        
        
            ## Warp Affine to reorient 
            input_pts = np.float32(corner_list)

            inputImg = cv2.circle(inputImg, (corner_list[0][0],corner_list[0][1]),5, (255,0,0),-1)
            # cv2.imshow('img',inputImg)

            output_pts = np.float32([[map_w,0],[0,0],[0,map_h],[map_w,map_h]])

            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

            warped_map = cv2.warpPerspective(inputImg, matrix, (map_w, map_h))
            # cv2.imshow('warped_map',warped_map)

            return warped_map

    else:
        print("Did not found 4 corners of map..")
        return None


def getAngle(a, b, c):
    '''Function to calculate angle between 3 points'''
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def get_mid_point(pt1,pt2):
    '''Function to return coordinates of midpoint of the two coordinate points'''
    x1,y1=pt1
    x2,y2=pt2
    midpnt = (int((x1+x2)/2), int((y1+y2)/2))
    return midpnt

def check_marker_pos(mapImg):
    '''Function takes in parameter as oriented map image.
        Returns marker_position and angle'''

    redPointerMask = getHSV_mask(mapImg,RED_POINTER_HSV_VALUES)

    ### Morphological closing operation performed on image using kernel 11x11 to fill the triangle 
    closing_kern_odd = 11
    kernel_closing = np.ones((closing_kern_odd,closing_kern_odd),np.uint8)
    redPointerMask = cv2.morphologyEx(redPointerMask, cv2.MORPH_CLOSE, kernel_closing)

    contours, heirarchy = cv2.findContours(redPointerMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxCnt = max(contours, key = cv2.contourArea)
    corner_list = extract_corners_list(maxCnt)  ## Get 3 corner coordinates of triangle(marker)
    
    dist_list = []
    for i in range(len(corner_list)):
        pt1 = corner_list[i]
        if i==2:
            pt2= corner_list[0]
        else:
            pt2= corner_list[i+1]
        
        dist = distance_between_points(pt1, pt2)
        dist_list.append(dist)
    
    ### Analyse which one is the shortest edge of the triangle and accordingly decide pointer nose and np.angle    
    shortest_dist = min(dist_list)
    index_min_dist = dist_list.index(shortest_dist)
    midpnt_smallest_side = []


    if index_min_dist == 0:
        pointer_nose = corner_list[2]
        midpnt_smallest_side = get_mid_point(corner_list[0],corner_list[1])
    elif index_min_dist== 1:
        pointer_nose = corner_list[0]
        midpnt_smallest_side = get_mid_point(corner_list[1],corner_list[2])
    elif index_min_dist ==2:
        pointer_nose = corner_list[1]
        midpnt_smallest_side = get_mid_point(corner_list[0],corner_list[2])


    imgH, imgW = mapImg.shape[:2]

    xpos = round(pointer_nose[0]/imgW,3)
    ypos = 1-round(pointer_nose[1]/imgH,3)  # Subtracted from 1 to get distance from bottom. 
    
    ### Take point in north from midpoint of smallest edge of marker
    vertical_point = (midpnt_smallest_side[0], midpnt_smallest_side[1]-50)
    hdg = getAngle(vertical_point, midpnt_smallest_side, pointer_nose)

    return [xpos, ypos, hdg]
    

def process_image(imgPath):
    '''Processes input image and returns the marker position and bearing'''

    img = cv2.imread(imgPath)
    
    ## Step1 : Extract and reorient map from the background
    
    blue_bg_mask = getHSV_mask(img, BLUE_BG_HSV_VALUES)
    map_mask = cv2.bitwise_not(blue_bg_mask)
    mapImage = reOrient_map(map_mask, img)

   
    ## Step2 : Calculate the position and angle of marker


    if mapImage is not None:
        xpos, ypos, hdg = check_marker_pos(mapImage)
        return [xpos, ypos, hdg]
        
    else:
        print("Error in orientation of map..")
        return None

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

print ("The filename to work on is %s." % sys.argv[1])

inputImgPath = sys.argv[1]

### Single function that will process all subprocesses and return output
list_output = process_image(inputImgPath)


if list_output is not None:
    xpos, ypos, hdg = list_output
        
    # Output the position and bearing in the form required by the test harness.
    print ("POSITION %.3f %.3f" % (xpos, ypos))
    print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

