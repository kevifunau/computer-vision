import cv2
import numpy as np
import sys
import os

def loadImage(im_address):
    # load image at greyscale mode
    image = cv2.imread(im_address, cv2.IMREAD_GRAYSCALE)
    assert (image is not None)
    return image

def getHistFrequency(image):

    # define 0-255 array
    hist_array = np.zeros(256, dtype=np.int32)
    for i in image.ravel():
        hist_array[i] += 1
    # transfer to frequency [0,1]
    hist_array_f = np.zeros(256, dtype=np.float)
    for i in range(256):
        hist_array_f[i] = hist_array[i] / float(image.size)
    return hist_array_f

def threshhold_otsu(hist_array_f):

    max_th = 0
    max_th_value = 0
    for t in range(256):
        # probability of foreground
        wf = sum([p for p in hist_array_f[:t + 1]])
        wb = sum([p for p in hist_array_f[t + 1:]])
        # varience
        vf = 0
        vb = 0
        for i in range(t + 1):
            vf += i * hist_array_f[i]
        for i in range(t + 1, 256):
            vb += i * hist_array_f[i]

        if (wf == 0 or wb == 0):
            continue

        vf = vf / wf
        vb = vb / wb

        # the inter-class varience
        r = wf * wb * (vf - vb) * (vf - vb)

        if r > max_th_value:
            max_th_value = r
            max_th = t

    return max_th

def getBinaryPng(image,threshold,output_image):
    # produce binary png
    image_c = image.copy()
    image_c = np.where(image_c > threshold, 255, 0)
    if os.path.exists(output_image):
        os.remove(output_image)
    cv2.imwrite(output_image, image_c)


if __name__ == "__main__":

    ### command line check
    if len(sys.argv) == 6 and sys.argv[1] == "--input" and sys.argv[3] == "--output" and sys.argv[5] =="--threshold":
        input_image = sys.argv[2]
        output_image = sys.argv[4]
    else:
        print("wrong input,sys out")
        sys.exit()


    # receive a greyscale image 2D-array by loading the png image
    image = loadImage(input_image)
    # find the frequency of each pixel (0-255)
    hist_array_f = getHistFrequency(image)
    # using otsu_algorithms to calc threshold
    threshold = threshhold_otsu(hist_array_f)
    print(threshold)
    # generate a binary image png image
    getBinaryPng(image,threshold,output_image)











