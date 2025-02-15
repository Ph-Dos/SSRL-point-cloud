import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def generateDepthMap():
    imgL = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoSGBM.create(
        numDisparities = 16 * 4,
        blockSize = 5 * 1,
    )
    disparityMap = stereo.compute(imgL,imgR)
    disparityMap = cv.medianBlur(disparityMap, 3)
    return disparityMap


def generatePointCloud(disparityMap):
    disparityMap = np.float32(np.divide(disparityMap, 16.0))

    # Baseline
    baseline = 178.089

    # Define the Q matrix using the intrinsic parameters and baseline
    Q = np.array([
        [1, 0, 0, -356.021],
        [0, -1, 0, 238.263],
        [0, 0, 0, 713.189],
        [0, 0, 1 / baseline, 0]
    ])

    imgL = cv.imread(sys.argv[1])

    projection = cv.reprojectImageTo3D(disparityMap, Q)
    maskMap = disparityMap > disparityMap.min()
    outputPoints = projection[maskMap]
    outputColors = imgL[maskMap]
    outputColors = outputColors.reshape(-1, 3)
    pointCloudFile(outputPoints, outputColors)


def pointCloudFile(vertices, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex {vertex_count}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open('out-put.ply', 'w') as f:
        f.write(ply_header.format(vertex_count=len(vertices)))
        vertices_colors = np.hstack([vertices, colors])
        np.savetxt(f, vertices_colors, fmt='%f %f %f %d %d %d')
    

def main():
    disparityMap = generateDepthMap()
    generatePointCloud(disparityMap)

main()
