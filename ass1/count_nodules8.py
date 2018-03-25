import cv2
import numpy as np
from collections import namedtuple, deque, Counter
import random
import os
import sys

#
# visit=[]
#
# def anyNeighbourhood(image,i,j):
#     r = []
#     h,w = image.shape
#     if i > 0 and image[i-1,j] ==0 and (i-1,j) not in visit :
#         visit.append((i-1,j))
#         r.append((i-1,j))
#     if i < h-1 and image[i+1,j] ==0 and (i+1,j) not in visit :
#         visit.append((i + 1, j))
#         r.append((i+1,j))
#     if j>0 and image[i,j-1]==0 and (i,j-1) not in visit :
#         visit.append((i, j-1))
#         r.append((i,j-1))
#     if j< w-1 and image[i,j+1]==0 and (i,j+1) not in visit:
#         visit.append((i, j+1))
#         r.append((i,j+1))
#     return r
# def floodfil(image,size):
#
#     h,w = image.B.shape
#     count_area = 0
#
#     for i in range(h):
#         for j in range(w):
#             # background or already labelled
#             if image.B[i,j] != 0:
#                 continue
#             else:
#                 count_area+=1
#                 col_list=[]
#
#                 # random a color for new component
#                 color_b = random.randint(1, 254)
#                 color_g = random.randint(1, 254)
#                 color_r = random.randint(1, 254)
#
#                 # new a queue
#                 _quene = deque()
#                 _quene.append((i,j))
#                 # start BFS search
#                 while _quene:
#                     x,y = _quene.popleft()
#                     col_list.append((x,y))
#                     # label the outed-queue point
#                     image.B[x,y] = color_b
#                     image.G[x,y] = color_g
#                     image.R[x,y] = color_r
#
#                     # get unlabelled neighbourhoods
#                     nei = anyNeighbourhood(image.B,x,y)
#                     # put in queue
#                     for ii,jj in nei:
#                         _quene.append((ii,jj))
#
#                 if len(col_list) < size:
#                     count_area-=1
#                     for _x,_y in col_list:
#                         # color with 255
#                         image.B[_x, _y] = 0
#                         image.G[_x, _y] = 0
#                         image.R[_x, _y] = 0
#     return count_area


CONNECT = {}
LABEL = 0


def isVaild(image, x, y):
    return label_arr[x, y] if image[x, y] != 255 else False


def Connectbase(x):
    global CONNECT
    return x if CONNECT[x] == x else Connectbase(CONNECT[x])


def Labelcheck(image, x, y):
    global LABEL
    global CONNECT


    w,nw,n,ne = 0,0,0,0

    if y > 0:
        w = isVaild(image, x, y - 1)

    if x>0 and y>0:
        nw = isVaild(image,x-1,y-1)

    if x > 0:
        n = isVaild(image, x - 1, y)

    if x>0 and y< image.shape[1]-1:
        ne = isVaild(image,x-1,y+1)


    if w ==False and nw == False and n == False and ne ==False:
        LABEL += 1
        CONNECT[LABEL] = LABEL
        return LABEL

    else:
        l = [w,nw,n,ne]
        l_noF = [ i for i in l if i != False]
        if set(l_noF) == 1:
            return min(l_noF)
        else:
            _min = min([Connectbase(i) for i in l_noF])
            for e in l_noF:
                CONNECT[e] = _min
            return _min


def Two_pass(image, size):
    global label_arr

    h, w = image.B.shape
    # first pass
    for i in range(h):
        for j in range(w):

            if image.B[i, j] != 255:
                # pixel is not background
                # do label check algo to get label value
                label_arr[i, j] = Labelcheck(image.B, i, j)
            else:
                continue

    # second pass
    for i in range(h):
        for j in range(w):
            # second pass
            if image.B[i, j] != 255:
                # pixel is not background
                label_arr[i, j] = CONNECT[label_arr[i, j]]
            else:
                continue

    areaLessArea = []
    for k, v in Counter(label_arr.ravel()).items():
        if v <= size:
            areaLessArea.append(k)

    # color
    for i in range(h):
        for j in range(w):

            if image.B[i, j] != 255 and label_arr[i, j] not in areaLessArea:

                random.seed(label_arr[i, j])

                color_b = random.randint(1, 254)
                color_g = random.randint(1, 254)
                color_r = random.randint(1, 254)

                image.B[i, j] = color_b
                image.G[i, j] = color_g
                image.R[i, j] = color_r

            else:
                continue


    return len(set(label_arr.ravel())) - 1 - len(areaLessArea)


if __name__ == "__main__":

    ## command line check
    if len(sys.argv) == 5 and sys.argv[1] == "--input" and sys.argv[3] == "--size" and sys.argv[4].isdigit():
        input_image = sys.argv[2]
        size = int(sys.argv[4])
        output_image = None
    elif len(sys.argv) == 7 and sys.argv[1] == "--input" and sys.argv[3] == "--size" and sys.argv[4].isdigit() and \
            sys.argv[5] == '--optional_output':
        input_image = sys.argv[2]
        size = int(sys.argv[4])
        output_image = sys.argv[6]
    else:
        print("wrong input,sys out")
        sys.exit()

    im = cv2.imread(input_image)
    Image = namedtuple("Image", ["B", "G", "R"])
    image = Image(im[:, :, 0], im[:, :, 1], im[:, :, 2])

    label_arr = np.zeros(image.B.shape)
    count_nodule = Two_pass(image, size)
    print(count_nodule)

    if output_image:
        if os.path.exists(output_image):
            os.remove(output_image)
        cv2.imwrite(output_image, im)



