import cv2
import numpy as np
#import pandas as pd
#import argparse

vid =cv2.VideoCapture(0)

def parFony(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",600,400)
cv2.createTrackbar("HUE Min","HSV",0,179, parFony)
cv2.createTrackbar("HUE Max","HSV",179,179, parFony)
cv2.createTrackbar("SAT Min","HSV",0,255, parFony)
cv2.createTrackbar("SAT Max","HSV",255,255, parFony)
cv2.createTrackbar("VALUE Min","HSV",0,255, parFony)
cv2.createTrackbar("VALUE Max","HSV",255,255, parFony)

while(1):
    ret, frame = vid.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    h_min=cv2.getTrackbarPos("HUE Min","HSV")
    h_max=cv2.getTrackbarPos("HUE Max","HSV")
    s_min=cv2.getTrackbarPos("SAT Min","HSV")
    s_max=cv2.getTrackbarPos("SAT Max","HSV")
    v_min=cv2.getTrackbarPos("VALUE Min","HSV")
    v_max=cv2.getTrackbarPos("VALUE Max","HSV")
    
    lower=np.array([h_min,s_min,v_min])
    higher=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(hsvFrame,lower,higher)
    result=cv2.bitwise_and(frame, frame, mask = mask)
    hStack=np.hstack([frame,result])
    
    
    if ret:
        #cv2.imshow("original",frame) 
        #cv2.imshow("HSVcapture",hsvFrame) 
        cv2.imshow("MaskCapture",mask)
        #cv2.imshow("MaskResultCapture",result)
        #cv2.imshow("HS",hStack)
        
    
    red_lower= np.array([136, 87, 111], np.uint8)
    red_upper= np.array([180, 255, 255], np.uint8)
    red_mask= cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower= np.array([25, 52, 72], np.uint8)
    green_upper= np.array([102, 255, 255], np.uint8)
    green_mask= cv2.inRange(hsvFrame, green_lower, green_upper)
    
    blue_lower= np.array([94, 80, 2], np.uint8)
    blue_upper= np.array([120, 255, 255], np.uint8)
    blue_mask= cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    kernal = np.ones((5,5), "uint8")
    
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame, mask = red_mask)
    
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(frame, frame, mask = green_mask)
    
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
    
    contours, heirarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 0, 255), 2)
            
            cv2.putText(frame, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    
    contours, heirarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
            
            cv2.putText(frame, "Green Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0)) 
    
    contours, heirarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 2)
            
            cv2.putText(frame, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0)) 
            
    cv2.imshow("HSV", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.videoCapture.release()
        cv2.destroyAllWindows()
        break
            

    
vid.release()
cv2.destroyAllWindows()

    
    