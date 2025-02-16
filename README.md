# Functionality

#### 1️⃣ Disparity Map Generation
* Disparity map is generated with Simple Stero
  * Disparity values are computed using triangulation of corresponding points determined by pixel positions from left and right images
  * Corresponding points are determined by locating the most similar pixel pattern using a kernel scanned over an epipolar line
      * Only works Because y-values in Left and Right images are Identical


#### 2️⃣ Point Cloud Generation
* Point cloud is generated using 3D rectification
  * Images provided are already rectified thus Q matrix can be composed from calibration data Via **`Data Set`** 


<hr>

### Virtual ENV Dependencies for WSL

1. ```python3 -m venv open3d_env source open3d_env/bin/activate```

2. ```pip install open3d numpy opencv-python```

[**`Data Set`**](https://vision.middlebury.edu/stereo/data/scenes2014/)
