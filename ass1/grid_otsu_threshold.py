import cv2
import numpy as np
import os
import sys
from math import sqrt



def loadImage(im_address):
    # load image at greyscale mode
    image = cv2.imread(im_address, cv2.IMREAD_GRAYSCALE)
    assert (image is not None)
    return image


def Imagesplit(image,grid_size):
    h,w = image.shape
    assert (grid_size < min(h,w))
    image_c = image.copy()

    for i in range(0,h,int(h/grid_size)):
        for j in range( 0,w,int(w/grid_size)):
            # do otsu_algo in all grids
            _grid = image[i:i+int(h/grid_size),j:j+int(w/grid_size)]
            _thr = threshhold_otsu(getHistFrequency(_grid))
            # generate grid binary png
            imgf = np.where(_grid>=_thr,255,0)

            # _thr,imgf= cv2.threshold(_grid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # put it in whole image
            image_c[i:i + int(h/grid_size), j:j + int(w/grid_size)] = imgf

    return image_c




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




if __name__ == "__main__":

    ### command line check
    if len(sys.argv) == 6 and sys.argv[1] == "--input" and sys.argv[4] == "--output" and sys.argv[3].isdigit():
        input_image = sys.argv[2]
        grid_size = int(sys.argv[3])
        output_image = sys.argv[5]
    else:
        print("wrong input,sys out")
        sys.exit()

    image = loadImage(input_image)
    image_new = Imagesplit(image,int(sqrt(grid_size)))


    #generate a binary image
    if os.path.exists(output_image):
        os.remove(output_image)
    cv2.imwrite(output_image, image_new)
















