import sys, os
import cv2
from skimage import filters
import numpy as np

def CalGradient(binimg):
    gradx = cv2.Sobel(binimg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grady = cv2.Sobel(binimg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    #subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradx, grady)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient

def Local_equalize(img, gridSize=5):
    calhe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(gridSize, gridSize))
    gray = calhe.apply(img)
    return gray

def stainSegmentation(stainfile, outpath):
    stainImg = cv2.imread(stainfile, -1)
    ### Compress staining image for registration
    tmp = stainImg.copy()
    height, width = stainImg.shape
    cmpImg = cv2.resize(tmp, (width // 4, height // 4), cv2.INTER_CUBIC)
    flip_img = np.rot90(cmpImg, 1)
    cv2.imwrite(os.path.join(outpath, 'HE_{0}_{1}_{2}_{3}.png'.format(1000, 1000, width, height)), flip_img)

    ### Create mask
    rotImg = cv2.flip(stainImg, 0)
    eql_img = Local_equalize(rotImg)
    gradient = CalGradient(eql_img)
    blurImg = cv2.GaussianBlur(gradient, (11, 11), 0)
    thresholds = filters.threshold_multiotsu(blurImg, classes=3)
    _, thresh = cv2.threshold(blurImg, thresholds[-1], 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    erodeimg = cv2.erode(closed, open_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilateimg = cv2.dilate(erodeimg, close_kernel, iterations=10)

    cnts, _ = cv2.findContours(dilateimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        mask = np.ones((height, width), np.uint8)
        conArea = height * width
        return conArea, mask

    contours = sorted(cnts, key=cv2.contourArea, reverse=True)
    c = contours[0]
    epsilon = 0.005*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    tmp = np.stack((eql_img, )*3, axis=2)
    cv2.drawContours(tmp, [approx], -1, (0, 0, 255), 10)
    cv2.imwrite(os.path.join(outpath, "Contour_staining_image.tif"), tmp)

    mask = np.zeros(stainImg.shape, np.uint8)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    cv2.fillPoly(mask, [c], 255)
    cv2.imwrite(os.path.join(outpath, "Staining_mask.tif"), mask)
    conArea = cv2.contourArea(c)
    return conArea, mask


stainfile = r"D:\xx\BGI\BrainReconstruction\ImageRegistration\Result\1.26.Register\registered_bx7.png"
outpath = r"D:\xx\BGI\TissueCutout\TestResult\1.26.test"
os.makedirs(outpath, exist_ok=True)
stainSegmentation(stainfile, outpath)