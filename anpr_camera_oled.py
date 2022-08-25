#!bin/python3

import sys
sys.path.append("/home/pi/Adafruit_Python_SSD1306")
import adafruit_ssd1306 as adafruit
import board, busio
from PIL import Image, ImageDraw, ImageFont
import os
import re

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit.SSD1306_I2C(128, 64, i2c)
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24)

oled.fill(255)
oled.show()

# import the necessary packages
import pytesseract
import numpy as np 
import imutils
import cv2
import traceback
from time import sleep
from picamera import PiCamera
from io import BytesIO
from itertools import repeat
from skimage.segmentation import clear_border

class PyImageSearchANPR:
    def __init__(self, minAR=3, maxAR=5.5, debug=False):
        self.minAR = minAR # min aspect ratio of plate 
        self.maxAR = maxAR # max aspect ratio of plate
        self.debug = debug

    # show image in debug window
    def debug_imshow(self, title, image, waitKey=False):
        if self.debug:
            cv2.imshow(title, image)
            
            cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a tophat morphological operation that will allow
        # us to reveal light regions (i.e., text) on dark backgrounds
        # (i.e., the license plate itself) 
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (78, 20))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
        if DETAILS: self.debug_imshow("Tophat", tophat)

        # next, find regions in the image that are dark
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
        dark = cv2.morphologyEx(gray, cv2.MORPH_OPEN, squareKern)
        dark = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if DETAILS: self.debug_imshow("thresh", dark)

        # compute the Scharr gradient representation of the topat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        #for i in range(3, -2, -2):
        i = -1
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=i)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        if DETAILS: self.debug_imshow(f"Scharr ksize-{i}", gradX)

        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if DETAILS: self.debug_imshow("Grad Thresh", thresh)

        # perform a series of erosions and dilations to clean up the
        # thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        if DETAILS: self.debug_imshow("Grad Erode/Dilate", thresh)
        
        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=dark)
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh)

        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnts = cnts[:keep]

        # return the list of contours
        return cnts, thresh
    
    def locate_license_plate(self, gray, candidates, thresh, clearBorder):
        lpCnt = None
        roi = None

        # loop over the license plate candidate contours
        for i, c in enumerate(candidates):
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            canvas = thresh[y:y + h, x:x + w]
            total = np.size(canvas)
            black = np.count_nonzero(canvas == 0)
            percentage = black/total*100

            self.debug_imshow(f"c{i} - {h}:{w} - {ar} - {percentage:.2f}%", canvas)
                
            # check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lpCnt = c
                licensePlate = gray[y-10:y+h+10, x-10:x+w+10 ]
                licensePlate = imutils.resize(licensePlate, height=100)
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                roi = clear_border(roi)
                dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
                dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                roi = (dist * 255).astype("uint8")
                roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                roi = cv2.erode(roi, None, iterations=2)
                roi = cv2.dilate(roi, None, iterations=1)
                roi = np.pad(roi, 10, mode="constant")

                self.debug_imshow("ROI cleaned", roi)

                break

        # return a 2-tuple of the license plate ROI and the contour
        # associated with it
        return (roi, lpCnt)
    
    def find_and_ocr(self, image, psm=7, clearBorder=True):
        self.debug_imshow("Original Image", image)
        # initialize the license plate text
        lpText = None
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (candidates, thresh) = self.locate_license_plate_candidates(gray)
        try:
            (lp, lpCnt) = self.locate_license_plate(gray, candidates, thresh, clearBorder=clearBorder)
        except ZeroDivisionError:
            lp = lpCnt = None
            
        # only OCR the license plate if the license plate ROI is not
        # empty
        if lp is not None:
            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)

        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return (lpText, lpCnt)
    
    def build_tesseract_options(self, psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

def speak(text):
    speakText = "".join([{
        "A": "A."
    }.get(char, char) + " " for char in text])
    
    os.system(f'espeak -s 125 -g 2 "{speakText}"')

def is_car_plate(text):
    if re.search("^\w{1,3}\d{1,4}$", text):
        return True
    return False

DEBUG = False
DETAILS = True

anpr = PyImageSearchANPR(debug=DEBUG)
anpr_strech = PyImageSearchANPR(minAR=4, maxAR=8)
countTotal = 0

oled.fill(0)
oled.show()

try:
    camera = PiCamera()
    camera.start_preview(alpha=0)#alpha=200)
    sleep(2)

    for stream in camera.capture_continuous(BytesIO(), format="jpeg"):
        print("captured ", end="")
        
        image = Image.open(stream).convert("RGB")
        cv_img = np.array(image)
        cv_img = cv_img[:, :, ::-1].copy()
        print("converted ", end="")

        countTotal += 1

        lpText, lpCont = anpr.find_and_ocr(cv_img, clearBorder=True)
        if lpText is not None:
            lpText = str.strip(lpText)
        else:
            lpText = ""
        
        print(f"Results: {lpText}")
        
        if is_car_plate(lpText):
            image = Image.new("1", (oled.width, oled.height))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,oled.width-1,oled.height-1), outline=255, fill=0, width=5)

            (fontW, fontH) = font.getsize(lpText)
            draw.text(
                ((oled.width-fontW)//2, (oled.height-fontH)//2),
                lpText,
                font=font,
                fill=255
            )
            
            oled.image(image)
            oled.show()
            
            speak(lpText)
        
        stream.seek(0)
        stream.truncate(0)
        
except Exception:
    print(traceback.format_exc())
    
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,oled.width-1,oled.height-1), outline=255, fill=0, width=5)

    (fontW, fontH) = font.getsize("Error!")
    draw.text(
        ((oled.width-fontW)//2, (oled.height-fontH)//2),
        "Error!",
        font=font,
        fill=255
    )

    oled.image(image)
    oled.show()
    sleep(2)
finally:
    print("done")
    camera.close()
    
    oled.fill(0)
    oled.show()
