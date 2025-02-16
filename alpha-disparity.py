import sys
import numpy as np
import cv2 as cv
import open3d as o3d


def generateDepthMap(SHOW_DEPTH):
    imgL = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoSGBM.create(
        numDisparities = 16 * 5,
        blockSize = 9,
    )
    disparityMap = stereo.compute(imgL,imgR)
    disparityMap = cv.normalize(disparityMap, disparityMap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparityMap = np.uint8(disparityMap)
    if SHOW_DEPTH:
        cv.imshow('Disparity Map', disparityMap)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return disparityMap


def generatePointCloud(disparityMap, fx, cx, cy, doffs, baseline):
    disparityMap = disparityMap.astype(np.float32)
    disparityMap[disparityMap == 0] = np.nan  # Avoid division by zero
    disparityMap /= 16.0  # Scale factor (if disparity is stored as fixed-point)

    # Compute 3D coordinates
    h, w = disparityMap.shape
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1 / baseline, doffs / baseline]
    ])

    # Reproject to 3D space
    points_3D = cv.reprojectImageTo3D(disparityMap, Q)

    # Mask out invalid points
    mask = (disparityMap > 0) & np.isfinite(points_3D[:,:,2])
    return points_3D[mask]


def main():
    fx = 999.421
    cx = 294.182
    cy = 252.932
    doffs = 32.778
    baseline = 193.001

    disparityMap = generateDepthMap(SHOW_DEPTH = True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(generatePointCloud(disparityMap, fx, cx, cy, doffs, baseline))

    # Save to PLY file
    o3d.io.write_point_cloud("out.ply", pcd)


main()
