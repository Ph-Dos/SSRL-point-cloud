import sys
import numpy as np
import cv2 as cv
import open3d as o3d


SHOW_DEPTH = False


def main():

    if len(sys.argv) < 3:
        print('Usage error: .py <Left.png> <Right.png> <calib.txt>')
        sys.exit(0)

    data = readData()
    focal = data['cam0'][0]
    principalX = data['cam0'][2]
    principalY = data['cam0'][5]
    dispOff = float(data['doffs'])
    baseline = float(data['baseline'])

    disparityMap = generateDepthMap(SHOW_DEPTH)
    points3D = generatePointCloud(disparityMap, focal, principalX, principalY, dispOff, baseline)
    pCloud = o3d.geometry.PointCloud()
    pCloud.points = o3d.utility.Vector3dVector(points3D)

    # Save to PLY file
    o3d.io.write_point_cloud("out.ply", pCloud)


# Computes and optionally displays a disparity map from two grayscale images
def generateDepthMap(SHOW_DEPTH):
    imgL = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoSGBM.create(
        numDisparities = 16 * 4,
        blockSize = 7,
        speckleWindowSize = 0,
        mode = cv.STEREO_SGBM_MODE_HH
    )
    
    # Image Processing to Increase disparityMap quality
    disparityMap = stereo.compute(imgL,imgR)
    disparityMap = cv.normalize(disparityMap, disparityMap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparityMap = cv.medianBlur(disparityMap, 5)

    if SHOW_DEPTH:
        disparityMap = np.uint8(disparityMap)
        cv.imshow('Disparity Map', disparityMap)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return disparityMap


# Converts a disparity map into a 3D point cloud using given camera parameters.
def generatePointCloud(disparityMap, focal, principalX, principalY, dispOff, baseline):
    disparityMap = disparityMap.astype(np.float32)

    #Prevent divide by zero
    disparityMap[disparityMap == 0] = np.nan  
    
    #requirement of openCV documentation
    disparityMap /= 16.0  

    # Compute 3D coordinates
    h, w = disparityMap.shape
    Q = np.array([
        [1, 0, 0, -principalX],
        [0, 1, 0, -principalY],
        [0, 0, 0, focal],
        [0, 0, -1 / baseline, dispOff / baseline]
    ])

    # Reproject to 3D space
    points3D = cv.reprojectImageTo3D(disparityMap, Q)

    # Mask out invalid points
    mask = (disparityMap > 0) & np.isfinite(points3D[:,:,2])
    return points3D[mask]


def readData():
    data = {}
    with open(sys.argv[3], 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            data[key] = value
    data['cam0'] = np.array([float(x) for x in data['cam0'].split()]) 
    return data


main()
