from __future__ import print_function  #
import cv2
import argparse
import os
from collections import namedtuple
import numpy as np

RANSAC_ITER = 500
RANSAC_THERSHOLD = 5.0
NFEATURE = 200

#######help function ##########
def myKnnMatch(des1,des2,k):
    match_list =[]
    for i in range(len(des1)):
        match_pair= sorted([ (i,j,np.linalg.norm(des2[j]-des1[i])) for j in range(len(des2))],key=lambda x:x[2])
        DMatch_list=[]
        for x,y,z in match_pair[:k]:
            DMatch_list.append(cv2.DMatch(x,y,0,z))
        match_list.append(DMatch_list)
    return match_list

def random_partition(data):
    '''
    randomly select two points
    '''
    np.random.shuffle(data)
    twoPoints = data[:4]
    samplePoint = data[4:]
    return twoPoints,samplePoint


def H_from_points(fp,tp):
    """ DLT algo"""
    assert fp.shape == tp.shape
    h = fp.shape[0]
    A = np.zeros((2*h,9))
    for i in range(h):
        x,y,u,v=fp[i,0],fp[i,1],tp[i,0],tp[i,1]
        A[2*i]= np.array([x,y,1,0,0,0,-u*x,-u*y,-u])
        A[2*i+1]=np.array([0,0,0,x,y,1,-v*x,-v*y,-v])
    U,D,V = np.linalg.svd(A)
    H=V[8].reshape(3,3)
    return H/H[-1,-1]



def conliner4(a):
    p1,p2,p3,p4=a
    return conliner3(p1,p2,p3) or conliner3(p1,p2,p4) or conliner3(p2,p3,p4)


def conliner3(p1,p2,p3):
    x1,y1=p1
    x2,y2=p2
    x3,y3=p3
    return abs((y2-y1)*(x3-x1)-(y3-y1)*(x2-x1)) < 1e-12

def homo_transform(pt,H):
    _pt = np.append(pt,1)
    _rs= np.dot(H,_pt)
    _rs/=_rs[-1]
    return _rs[:2]


def myfindHomo_ransac(src_pts,dst_pts):
    comb = list(zip(src_pts,dst_pts))
    good_inlier_mask = None
    best_inlier=-1

    #find good_mask and inlier
    for _ in range(RANSAC_ITER):
        a,b = random_partition(comb)
        while conliner4([ pair[0] for pair in a]):
            a,b = random_partition(comb)
        mask = H_from_points(np.array([pair[0] for pair in a]),np.array([pair[1] for pair in a] ))

        _inlier=0
        for src,dst in b:
            trans_dst = homo_transform(src,mask)
            diff = np.sqrt(sum((dst-trans_dst)**2))
            if diff <=RANSAC_THERSHOLD:
                _inlier +=1
        if _inlier > best_inlier:
            good_inlier_mask = mask
            best_inlier = _inlier

    inlier_src=[]
    inlier_dst=[]
    for src,dst in comb:
            trans_dst = homo_transform(src,good_inlier_mask)
            diff = np.sqrt(sum((dst-trans_dst)**2))
            if diff <=RANSAC_THERSHOLD:
                inlier_src.append(src)
                inlier_dst.append(dst)

    # find the best_mask
    best_mask=[]
    best_diff=np.float("Inf")
    comb = list(zip(inlier_src,inlier_dst))
    for _ in range(RANSAC_ITER):
        a,b = random_partition(comb)
        while conliner4([ pair[0] for pair in a]):
            a,b = random_partition(comb)
        mask = H_from_points(np.array([pair[0] for pair in a]),np.array([pair[1] for pair in a] ))

        diff=0
        for src,dst in b:
            trans_dst = homo_transform(src,mask)
            diff += np.sqrt(sum((dst-trans_dst)**2))
        if diff < best_diff:
            best_diff = diff
            best_mask = mask

    return best_mask

def mywarpPerspective(img, M, size):
    new_img = np.zeros([size[1], size[0], 3], dtype=np.uint8)

    for i in range(size[1]):
        for j in range(size[0]):
            new_x = np.divide(M[1, 0] * j + M[1, 1] * i + M[1, 2], M[2, 0] * j + M[2, 1] * i + M[2, 2])
            new_y = np.divide(M[0, 0] * j + M[0, 1] * i + M[0, 2], M[2, 0] * j + M[2, 1] * i + M[2, 2])
            new_x = int(round(new_x))
            new_y = int(round(new_y))
            if 0 <= new_x < size[1] and 0 <= new_y < size[0]:
                new_img[new_x][new_y] = img[i][j]

    return new_img


def getHomoMatrix(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURE)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, 2)
    good_matchlist=[m for m,n in matches if  m.distance < 0.7 * n.distance]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matchlist]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matchlist]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    return H



########step 1############
def up_to_step_1(imgs):
    modified_imgs=[]
    for imname,img in imgs:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURE)
        kps = sift.detect(gray_img, None)
        modified_img = cv2.drawKeypoints(img, kps, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        modified_imgs.append((imname,modified_img))
    return modified_imgs

def save_step_1(imgs, output_path='./output/step1'):

    if os.path.exists(output_path) != True:
        os.mkdir(output_path)

    for name,img in imgs:
        cv2.imwrite(output_path+"/{}.jpg".format(name),img)


########step 2 ###########
def up_to_step_2(imgs):
    Image_pair = namedtuple("Image_pair",["query","train"])
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURE)
    match_imgs=[]
    while len(imgs) != 1:
        pair = Image_pair(imgs.pop(0),imgs[0])
        kps1, des1 = sift.detectAndCompute(pair.query[1], None)
        kps2, des2 = sift.detectAndCompute(pair.train[1], None)
        raw_match_list = myKnnMatch(des1,des2,2)
        good_match_list = [[m]for m, n in raw_match_list if m.distance < 0.75 * n.distance]
        match_img = cv2.drawMatchesKnn(pair.query[1],kps1,pair.train[1],kps2,good_match_list,None,matchColor=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        new_image_name = pair.query[0] + "_" + str(len(kps1)) + "_" + pair.train[0] + "_" + str(len(kps2)) + "_" + str(len(good_match_list))
        match_imgs.append((new_image_name,match_img))
    return match_imgs

def save_step_2(imgs, output_path="./output/step2"):

    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    for name,img in imgs:
        cv2.imwrite(output_path+"/{}.jpg".format(name),img)

########step 3 ###########

def up_to_step_3(imgs):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURE)
    while len(imgs)!=1:
        img1_name,img1 = imgs.pop(0)
        img2_name,img2 = imgs[0]
        kps1, des1 = sift.detectAndCompute(img1, None)
        kps2, des2 = sift.detectAndCompute(img2, None)
        raw_match_list_lr = myKnnMatch(des1, des2, 2)
        raw_match_list_rl = myKnnMatch(des2, des1, 2)
        good_match_list_lr = [m for m, n in raw_match_list_lr if m.distance < 0.75 * n.distance]
        good_match_list_rl = [m for m, n in raw_match_list_rl if m.distance < 0.75 * n.distance]

        # lr
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_match_list_lr])
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_match_list_lr])
        M= myfindHomo_ransac(src_pts,dst_pts)
        size = img1.shape
        new_img1 = mywarpPerspective(img1,M,(size[1],size[0]))
        save_step_3((img1_name,img2_name,new_img1))

        # rl
        src_pts = np.float32([kps2[m.queryIdx].pt for m in good_match_list_rl])
        dst_pts = np.float32([kps1[m.trainIdx].pt for m in good_match_list_rl])
        M = myfindHomo_ransac(src_pts, dst_pts)
        size = img2.shape
        new_img2 = mywarpPerspective(img2, M, (size[1], size[0]))
        save_step_3((img2_name,img1_name,new_img2))

def save_step_3(img_pairs, output_path="./output/step3"):

    if  os.path.exists(output_path) != True:
        os.mkdir(output_path)
    img1_name,img2_name,img = img_pairs
    cv2.imwrite("{}/warped_{}_ref_{}.jpg".format(output_path, img1_name, img2_name), img)

########step 4 ###########

def CoorTrans(img, M):
    min_x, min_y, max_x, max_y = float('Inf'), float('Inf'), float("-Inf"), float("-Inf")
    (h1, w1) = img.shape[:2]
    top_left = homo_transform([0, 0], M)
    top_right = homo_transform([w1, 0], M)
    bottom_left = homo_transform([0, h1], M)
    bottom_right = homo_transform([w1, h1], M)

    for corner in [top_left, top_right, bottom_left, bottom_right]:
        a, b = corner[0], corner[1]
        min_x = a if a < min_x else min_x
        max_x = a if a > max_x else max_x
        min_y = b if b < min_y else min_y
        max_y = b if b > max_y else max_y
    T = np.zeros(shape=(3, 3))
    T[0] = [1, 0, 0]
    T[1] = [0, 1, 0]
    T[2] = [-min_x, -min_y, 1]

    return int(min_x), int(max_x), int(min_y), int(max_y), T.transpose()


def merge_images_translation(image1, image2, offset):
    ## Put images side-by-side into 'image'.
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    (ox, oy) = offset
    ox = int(ox)
    oy = int(oy)
    oy = 0

    image = np.zeros((h1 + oy, w1 + ox, 3), np.uint8)

    image[:h1, :w1] = image1
    image[:h2, ox:ox + w2] = image2

    return image



def up_to_step_4(imgs,output_path):
    imgs=imgs[:5]

    left_image = imgs[0]
    assert  left_image is not None
    for i in range(1,len(imgs)):
        right_image = imgs[i]
        assert right_image is not None
        mask = getHomoMatrix(left_image,right_image)
        minx,max_x,miny,max_y,T = CoorTrans(left_image,mask)
        H = np.dot(T,mask)
        offset_x,offset_y = abs(minx),abs(miny)
        left_image = cv2.warpPerspective(left_image,H,(abs(max_x-minx),abs(max_y-miny)))
        diff_y = offset_y + right_image.shape[0]  - left_image.shape[0]
        diff_x = offset_x + right_image.shape[1] -left_image.shape[1]
        if diff_y>=0:
            blank = np.zeros(left_image.shape,np.uint8)
            left_image = np.concatenate((left_image,blank),axis=0)
        if diff_x>=0:
            blank = np.zeros(left_image.shape,np.uint8)
            left_image = np.concatenate((left_image,blank),axis=1)
        for i in range(offset_y,right_image.shape[0]+offset_y):
            for j in range(offset_x,right_image.shape[1] + offset_x):
                left_image[i][j] = right_image[i-offset_y][j-offset_x]

    save_step_4(left_image,output_path)


def save_step_4(img, output_path="./output/step4"):

    if  os.path.exists(output_path) != True:
        os.mkdir(output_path)
    cv2.imwrite("{}/stitch.jpg".format(output_path), img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )
    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )
    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )
    args = parser.parse_args()

    imname_imgs=[]
    filename_list = os.listdir(args.input)
    filename_list.sort()

    for filename in filename_list:
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        if img is None:
            raise Exception("image load failure!")
        imname_imgs.append((filename.split(".")[0],img))

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imname_imgs)
        save_step_1(modified_imgs, args.output)

    elif args.step == 2:
        print("Running step 2")
        modified_imgs = up_to_step_2(imname_imgs)
        save_step_2(modified_imgs, args.output)

    elif args.step == 3:
        print("Running step 3")
        up_to_step_3(imname_imgs)
    elif args.step == 4:
        print("Running step 4")
        imgs = [y for x,y in imname_imgs]
        panoramic_img = up_to_step_4(imgs,args.output)




