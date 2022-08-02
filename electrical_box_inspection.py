from os import system
import math
import cv2
import numpy as np

#system('clear')
print("This is a script to test the functionality of Spot's box inspection.\n")
print("Inspection will now begin with the given box.\n")

box_num = input("Pick a box (1-9): ")
                                                 #         Presets         Brightness     Breaker Thresh     Fuse Pixel Ratio

box = '../images/Box_Examples/box_1.jpg'         # (On,  3 Fuses, 28/30)     (109)           (120-180)            (1.52)
if box_num == '2': 
    box = '../images/Box_Examples/box_2.jpg'     # (Off, 3 Fuses, 28/30)     (108)           (150-190)            (1.35)
# elif box_num == '3': 
#     box = '../images/Box_Examples/box_3.jpg'
elif box_num == '4': 
    box = '../images/Box_Examples/box_4.jpg'     # (Off, 1 Fuses, 28/30)     (110)           (130-180)
elif box_num == '5': 
    box = '../images/Box_Examples/box_5.jpg'     # (Off, 0 Fuses, 28/30)     (111)           (130-190)
elif box_num == '6': 
    box = '../images/Box_Examples/box_6.jpg'     # (Off, 3 Fuses, 60/0)      (125)           (210-240)            (2.12)
elif box_num == '7': 
    box = '../images/Box_Examples/box_7.jpg'     # (Off, 3 Fuses, 40/10)     (124)           (220-240)            (2.1)
# elif box_num == '8':                          
#     box = '../images/Box_Examples/box_8.jpg'   # Your example

box = cv2.imread(box)
brightness = round(np.average(box))
# print(brightness)

if 105 < brightness and brightness <= 115:
    breaker_reference = '../images/Box_Parts/breaker_1.jpg'
    fuse_reference = '../images/Box_Parts/fuses_1.jpg'
    temp_reference = '../images/Box_Parts/tempbox_1.jpg'
elif 115 < brightness and brightness <= 125:
    breaker_reference = '../images/Box_Parts/breaker_2.jpg'
    fuse_reference = '../images/Box_Parts/fuses_2.jpg'
    temp_reference = '../images/Box_Parts/tempbox_2.jpg'
# elif 'min_bright' < brightness and brightness = 'max_bright:      # Your example
#     breaker_reference = '../images/Box_Parts/breaker_3.jpg'
#     fuse_reference = '../images/Box_Parts/fuses_3.jpg'
#     temp_reference = '../images/Box_Parts/tempbox_3.jpg'

switch_min_threshold = 4.5 * brightness - 330
fuse_pixel_ratio = .045 * brightness - 3
    
print("------------------------------------------------------\n")
print("Finding breaker...\n")

img = box[box.shape[0]//2:box.shape[0], 0:box.shape[1]//2 ] #crop to the lower left quadrant
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
breaker = cv2.imread(breaker_reference)
breaker_gray = cv2.cvtColor(breaker, cv2.COLOR_RGB2GRAY)

#Apply feature deaction between robot image and caparable image
orb = cv2.ORB_create(nfeatures = 1000)
img_preview = np.copy(img)
breaker_preview = np.copy(breaker)
dots = np.copy(breaker)
train_keypoints, train_descriptor = orb.detectAndCompute(breaker_gray, None)
test_keypoints, test_descriptor = orb.detectAndCompute(img_gray, None)
cv2.drawKeypoints(breaker, train_keypoints, breaker_preview, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(breaker, train_keypoints, dots, flags=2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(train_descriptor, test_descriptor)
matches = sorted(matches, key = lambda x : x.distance)
numMatches = 300
good_matches = matches[:numMatches]
train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)
h,w = breaker_gray.shape[:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
dots = cv2.drawMatches(dots,train_keypoints,img,test_keypoints,good_matches, None,flags=2)
result = cv2.polylines(img, [np.int32(dst)], True, (0,255,0), 10, cv2.LINE_AA)
# cv2.imshow('Matches', cv2.resize(dots,(dots.shape[1]//3, dots.shape[0]//3)))
# cv2.waitKey()

#Crop boxed image to a rectangle
if int(dst[0][0][0]) < 0: start_col = 0 
else: start_col =  int(dst[0][0][0])
if int(dst[2][0][0]) < 0: end_col = 0
else: end_col =  int(dst[2][0][0])
if int(dst[0][0][1]) < 0: start_row = 0
else: start_row =  int(dst[0][0][1])
if int(dst[2][0][1]) < 0: end_row = 0
else: end_row =  int(dst[2][0][1])
cropped = img_preview[start_row:end_row, start_col:end_col]

try:
    cv2.imshow('component_outline', cv2.resize(box,(box.shape[1]//5, box.shape[0]//5)))
    print("Breaker identified.\n")
except:
    print("Breaker failed to be identified. Ending script.\n")
    exit()
cv2.waitKey()

print("Finding switch...\n")

switch_cropped = cropped[cropped.shape[0]*8//18:cropped.shape[0]*12//18, cropped.shape[1]*10//20:cropped.shape[1]*13//20 ]
cv2.imshow('switch_outline.jpg', cv2.resize(switch_cropped,(switch_cropped.shape[1]//2, switch_cropped.shape[0]//2)))
print("Switch identified.\n")
cv2.waitKey()

print("Determining position of breaker...\n")

img = switch_cropped
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, threshold = cv2.threshold(gray, switch_min_threshold, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', threshold)
cv2.waitKey()
kernel = np.ones((20,20), np.uint8) #Number enhances splotches
img_dilation = cv2.dilate(threshold, kernel, iterations=1)
cv2.imshow('dil', img_dilation)
cv2.waitKey()
kernel = np.ones((25,25), np.uint8) #Number enhances splotches
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
cv2.imshow('erode', img_erosion)
cv2.waitKey()
contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 3 or len(contours) < 2:
    print('Too much interference to identify state of breaker')
    exit()
else:
    (x, y_off, w, h_off) = cv2.boundingRect(contours[0])
    cv2.rectangle(img, (x, y_off), (x+w, y_off+h_off), (0, 255, 0), 2)
    off_center = y_off + h_off/2
    
    (x, y_15, w, h_15) = cv2.boundingRect(contours[1])
    cv2.rectangle(img, (x, y_15), (x+w, y_15+h_15), (0, 255, 0), 2)
    switch_center = y_15 + h_15/2
    
    if len(contours) == 3:
        (x, y_on, w, h_on) = cv2.boundingRect(contours[2]) 
        cv2.rectangle(img, (x, y_on), (x+w, y_on+h_on), (0, 255, 0), 2)
        on_center = y_on + h_on/2

        #Compares the distances of the boxes to determine whether the switch is closer to the 'ON' or 'OFF' position
        if (switch_center - on_center < off_center - switch_center):
            print('Breaker is in "ON" position.')
        else:
            print('Breaker is in "OFF" position.')
    else:
        print('Breaker is in "OFF" position.')

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
print("------------------------------------------------------\n")
print('Finding fuses...\n')
# quit()
img = box[box.shape[0]//4:box.shape[0]//4*3, box.shape[1]//2:box.shape[1] ]
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fuses = cv2.imread(fuse_reference)
fuses_gray = cv2.cvtColor(fuses, cv2.COLOR_RGB2GRAY)

#Create a border around fuses
orb = cv2.ORB_create(nfeatures = 1000)
img_preview = np.copy(img)
fuses_preview = np.copy(fuses)
dots = np.copy(fuses)
train_keypoints, train_descriptor = orb.detectAndCompute(fuses_gray, None)
test_keypoints, test_descriptor = orb.detectAndCompute(img_gray, None)
cv2.drawKeypoints(fuses, train_keypoints, fuses_preview, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(fuses, train_keypoints, dots, flags=2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(train_descriptor, test_descriptor)
matches = sorted(matches, key = lambda x : x.distance)
numMatches = 300
good_matches = matches[:numMatches]
train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)
h,w = fuses_gray.shape[:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
dots = cv2.drawMatches(dots,train_keypoints,img,test_keypoints,good_matches, None,flags=2)
result = cv2.polylines(img, [np.int32(dst)], True, (0,255,0), 10, cv2.LINE_AA)

#crop border to just fuse box
if int(dst[0][0][0]) < 0: start_col = 0
else: start_col =  int(dst[0][0][0])
if int(dst[2][0][0]) < 0: end_col = 0
else: end_col =  int(dst[2][0][0])
if int(dst[0][0][1]) < 0: start_row = 0
else: start_row =  int(dst[0][0][1])
if int(dst[2][0][1]) < 0: end_row = 0
else: end_row =  int(dst[2][0][1])
cropped = img_preview[start_row:end_row, start_col:end_col]

try:
    cv2.imshow('component_outline', cv2.resize(box,(box.shape[1]//5, box.shape[0]//5)))
    print("Fuses identified.\n")
except:
    print("Breaker failed to be identified. Fuses likely missing. Ending script.\n")
    exit()
cv2.waitKey()

print("Determining status of fuses...\n")

image = cropped
try:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    #Displays the number of white to black pixels
    numWhite = np.sum(segmented_image > 150)
    numBlack = np.sum(segmented_image < 100)
    #print('Black/White pixel ratio: ~', round(numBlack/numWhite, 2))

    #Compares the normal ratio of black/white pixels to the image to see if a fuse is missing
    if (numBlack/numWhite > fuse_pixel_ratio): 
        print("Test FAILED: One or more fuses is missing.\n")
    else:
        print("Test PASSED: All fuses seem to be intact.\n")

    cv2.imshow('Black/white', cv2.resize(segmented_image,(segmented_image.shape[1]//2, segmented_image.shape[0]//2)))
except:
    print("Test FAILED: Possible all fuses are missing.\n")

cv2.waitKey()
cv2.destroyAllWindows()
print("------------------------------------------------------\n")
print("Finding temperature box...\n")
# quit()
img = box[box.shape[0]*4//24:box.shape[0]*9//24, box.shape[1]*8//24:box.shape[1]*13//24 ]
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
reference =  cv2.imread(temp_reference)
reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(nfeatures = 1000)
img_preview = np.copy(img)
reference_preview = np.copy(reference)
dots = np.copy(reference)
train_keypoints, train_descriptor = orb.detectAndCompute(reference_gray, None)
test_keypoints, test_descriptor = orb.detectAndCompute(img_gray, None)
cv2.drawKeypoints(reference, train_keypoints, reference_preview, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(reference, train_keypoints, dots, flags=2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(train_descriptor, test_descriptor)
matches = sorted(matches, key = lambda x : x.distance)
numMatches = 300
good_matches = matches[:numMatches]
train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)
h,w = reference_gray.shape[:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M) 
dots = cv2.drawMatches(dots,train_keypoints,img,test_keypoints,good_matches, None,flags=2)
result = cv2.polylines(img, [np.int32(dst)], True, (0,255,0),10, cv2.LINE_AA)

if int(dst[0][0][0]) < 0: start_col = 0
else: start_col =  int(dst[0][0][0])
if int(dst[2][0][0]) < 0: end_col = 0
else: end_col =  int(dst[2][0][0])
if int(dst[0][0][1]) < 0: start_row = 0
else: start_row =  int(dst[0][0][1])
if int(dst[2][0][1]) < 0: end_row = 0
else: end_row =  int(dst[2][0][1])
cropped = img_preview[start_row:end_row, start_col:end_col]

try:
    cv2.imshow('component_outline', cv2.resize(box,(box.shape[1]//5, box.shape[0]//5)))
    print("Temperature box identified.\n")
except:
    print("Temperature box failed to be identified. Ending script.\n")
    exit()

cv2.waitKey()
print("Determining blue and red dial temperatures...\n")

#Inspect temp box

img = cropped

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
final_img = img.copy()
dial_radius = img.shape[1]*1//10
scale = 8

#Isolate red dial and circle it
mask_r1 = cv2.inRange(hsv, np.array([0,50,50]), np.array([10,255,255]))
mask_r2 = cv2.inRange(hsv, np.array([170,60,60]), np.array([180,255,255]))
mask_r = mask_r1 + mask_r2
res_r = cv2.bitwise_and(img, img, mask= mask_r)
# cv2.imshow('res_r', res_r)
# cv2.waitKey()
gray = cv2.cvtColor(res_r, cv2.COLOR_RGB2GRAY)
ret, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# cv2.imshow('red_threshold', threshold)
# cv2.waitKey()
kernal = 10,10
img_dilation = cv2.dilate(threshold, np.ones((kernal), np.uint8), 1)
# cv2.imshow('red_img_dilation', img_dilation)
# cv2.waitKey()
kernal = 30,30 
img_erosion = cv2.erode(img_dilation, np.ones((kernal), np.uint8), 1)
# cv2.imshow('red_img_erosion', img_erosion)
# cv2.waitKey()
contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(res_r, contours, -1, (0, 255, 0), 10)

#Find the center of the circle 
minX = contours[0][0][0][0]
maxX = contours[0][0][0][0]
minY = contours[0][0][0][1]
maxY = contours[0][0][0][1]
for i in range(len(contours[0])):
    if contours[0][i][0][0] < minX:
        minX = contours[0][i][0][0]
    if contours[0][i][0][0] > maxX:
        maxX = contours[0][i][0][0]
    if contours[0][i][0][1] < minY:
        minY = contours[0][i][0][1]
    if contours[0][i][0][1] > maxY:
        maxY = contours[0][i][0][1]
center_x = int((minX + maxX) / 2)
center_y = int((minY + maxY) / 2)
cv2.circle(final_img, (center_x, center_y), 2, (255, 0, 0), 2)
cv2.circle(final_img, (center_x, center_y), dial_radius, (0, 255, 0), 3)

cropped_red_dial = img[ center_y - dial_radius: center_y + dial_radius, center_x - dial_radius: center_x + dial_radius]
h, w, _ = cropped_red_dial.shape
cropped_red_dial = cv2.resize(cropped_red_dial, (w * scale, h * scale))
# cv2.imshow('cropped_red_dial', cropped_red_dial)
# cv2.waitKey()
red_gray = cv2.cvtColor(cropped_red_dial, cv2.COLOR_BGR2GRAY)
red_thresh = cv2.adaptiveThreshold(red_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 65, 5)
# cv2.imshow('red_thresh_1', red_thresh)
# cv2.waitKey()
red_mask = np.zeros_like(red_thresh)
cv2.circle(red_mask, (int(w * scale / 2), int(h * scale * 1/2)), int(w * scale *  1/3), 255, int(w * scale * 1/6))
red_thresh = cv2.bitwise_and(red_thresh, red_mask)
cv2.imshow("red_thresh_2", red_thresh)
cv2.waitKey()

contours, _ = cv2.findContours(red_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    m = max([(c, cv2.contourArea(c)) for c in contours], key=lambda i: i[1])[0]

    M = cv2.moments(m)

    if M['m00'] > 0:
        x = int(round(M['m10'] / M['m00']) / scale)
        y = int(round(M['m01'] / M['m00']) / scale)
        point_x = int(w / 2) - x
        point_y = int(h / 2) - y
        point_x = center_x - point_x
        point_y = center_y - point_y
        cv2.circle(final_img, (point_x, point_y), 2, (255, 0, 0), 2)

#Calculate the angle between the arrow and the center of the dial to determine the temperature
angle = math.atan2(center_y - point_y, (center_x - point_x) * -1) * (180/math.pi)
if angle < 0:
    angle = angle + 360

if -10 <= angle and angle <= 250:
    redTemp = int(-0.2 * abs(angle) + 48)
    print("Red dial's temperature: ~", redTemp, "\u00B0C.\n")
elif 290 <= angle and angle <= 350:
    redTemp = int(-0.2 * abs(300-angle) + 60)
    print("Red dial's temperature: ~", redTemp, "\u00B0C.\n")
else:
    print("The red dial's temperature reading has malfunctioned.\n")

#Isolate blue dial and circle it
mask_b = cv2.inRange(hsv, np.array([100,100,140]), np.array([170,170,255])) 
res_b = cv2.bitwise_and(img, img, mask= mask_b)
# cv2.imshow('res_b', res_b)
# cv2.waitKey()
gray = cv2.cvtColor(res_b, cv2.COLOR_RGB2GRAY)
ret, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
# cv2.imshow('blue_threshold', threshold)
# cv2.waitKey()
kernal = 20,20
img_dilation = cv2.dilate(threshold, np.ones((kernal), np.uint8), 1)
# cv2.imshow('blue_img_dilation', img_dilation)
# cv2.waitKey()
kernal = 30,30 
img_erosion = cv2.erode(img_dilation, np.ones((kernal), np.uint8), 1)
# cv2.imshow('blue_img_erosion', img_erosion)
# cv2.waitKey()
contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(res_b, contours, -1, (0, 255, 0), 5)

#Find the center of the circle 
minX = contours[0][0][0][0]
maxX = contours[0][0][0][0]
minY = contours[0][0][0][1]
maxY = contours[0][0][0][1]
for i in range(len(contours[0])):
    if contours[0][i][0][0] < minX:
        minX = contours[0][i][0][0]
    if contours[0][i][0][0] > maxX:
        maxX = contours[0][i][0][0]
    if contours[0][i][0][1] < minY:
        minY = contours[0][i][0][1]
    if contours[0][i][0][1] > maxY:
        maxY = contours[0][i][0][1]
center_x = int((minX + maxX) / 2)
center_y = int((minY + maxY) / 2)
cv2.circle(final_img, (center_x, center_y), 2, (0, 0, 255), 2)
cv2.circle(final_img, (center_x, center_y), dial_radius, (0, 255, 0), 3)

cropped_blue_dial = img[ center_y - dial_radius: center_y + dial_radius, center_x - dial_radius: center_x + dial_radius]
h, w, _ = cropped_blue_dial.shape
cropped_blue_dial = cv2.resize(cropped_blue_dial, (w * scale, h * scale))
# cv2.imshow('cropped_blue_dial', cropped_blue_dial)
# cv2.waitKey()
blue_gray = cv2.cvtColor(cropped_blue_dial, cv2.COLOR_BGR2GRAY)
blue_thresh = cv2.adaptiveThreshold(blue_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 65, 5)
# cv2.imshow('blue_thresh', blue_thresh)
# cv2.waitKey()
blue_mask = np.zeros_like(blue_thresh)
cv2.circle(blue_mask, (int(w * scale / 2), int(h * scale * 1/2)), int(w * scale *  1/3), 255, int(w * scale * 1/6))
blue_thresh = cv2.bitwise_and(blue_thresh, blue_mask)
cv2.imshow("blue_thresh_2", blue_thresh)
cv2.waitKey()

contours, _ = cv2.findContours(blue_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    m = max([(c, cv2.contourArea(c)) for c in contours], key=lambda i: i[1])[0]

    M = cv2.moments(m)

    if M['m00'] > 0:
        x = int(round(M['m10'] / M['m00']) / scale)
        y = int(round(M['m01'] / M['m00']) / scale)
        point_x = int(w / 2) - x
        point_y = int(h / 2) - y
        point_x = center_x - point_x
        point_y = center_y - point_y
        cv2.circle(final_img, (point_x, point_y), 2, (0, 0, 255), 2)

#Calculate the angle between the arrow and the center of the dial to determine the temperature
angle = math.atan2(center_y - point_y, (center_x - point_x) * -1) * (180/math.pi)
if angle < 0:
    angle = angle + 360

if -10 <= angle and angle <= 250:
    blueTemp = int(-0.2 * abs(angle) + 48)
    print("Blue dial's temperature: ~", blueTemp, "\u00B0C.\n")
elif 290 <= angle and angle <= 350:
    blueTemp = int(-0.2 * abs(300-angle) + 60)
    print("Blue dial's temperature: ~", blueTemp, "\u00B0C.\n")
else:
    print("The blue dial's temperature reading has malfunctioned.\n")

print("------------------------------------------------------\n")
print("All components found. Ending program.\n")

cv2.imshow('temp_box', final_img)
cv2.waitKey()
cv2.destroyAllWindows()