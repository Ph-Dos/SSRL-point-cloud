import sys
import numpy as np
import cv2 as cv

def generateDepthMap(SHOW_DEPTH):
    imgL = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoSGBM.create(
        numDisparities = 16 * 5,
        blockSize = 9,
    )
    disparityMap = stereo.compute(imgL,imgR)
    disparityMap = np.float32(disparityMap)
    disparityMap = cv.normalize(disparityMap, disparityMap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparityMap = np.uint8(disparityMap)
    disparityMap = cv.medianBlur(disparityMap, 3)

    if SHOW_DEPTH:
        cv.imshow('Disparity Map', disparityMap)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return disparityMap


def generatePointCloud(disparityMap):
    disparityMap = np.float32(np.divide(disparityMap, 16.0))

    Q = generateQMatrix()

    projection = cv.reprojectImageTo3D(disparityMap, Q, handleMissingValues = False)

    imgL = cv.imread(sys.argv[1])

    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    maskMap = disparityMap > disparityMap.min()
    outputPoints = projection[maskMap]
    outputColors = colors[maskMap]
    pointCloudFile(outputPoints, outputColors)


def generateQMatrix():
    calibData = {}
    with open(sys.argv[3], 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            calibData[key] = value
    P1 = np.array(list(map(float, calibData['cam0'].split()))).reshape(3, 3)
    P2 = np.array(list(map(float, calibData['cam1'].split()))).reshape(3, 3)
    Q = np.array([
        [1, 0, 0, -P1[0, 2]],
        [0, 1, 0, -P1[1, 2]],
        [0, 0, 0, -P1[0, 0]],
        [0, 0, -1/float(calibData['baseline']), 0]
    ])
    return Q


def pointCloudFile(vertices, colors):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])


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
        np.savetxt(f, vertices, fmt = '%f %f %f %d %d %d')


def main():
    disparityMap = generateDepthMap(SHOW_DEPTH = False)
    generatePointCloud(disparityMap)

main()
