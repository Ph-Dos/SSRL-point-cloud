# Functionality

#### 1️⃣ Disparity Map Generation
* Disparity map is generated with Simple Stero
  * Disparity values are computed using triangulation of corresponding points determined by
    * Intersection point of rays cast from corresponding pixels in left and right images
    * Difference between Left x-value and Right x-value or Baseline
  * Corresponding points are determined by locating the most similar pixel pattern using a kernel scanned over an epipolar line
      * Only works Because y-values in Left and Right images are Identical


#### 2️⃣ Point Cloud Generation
* Point cloud is generated using 3D reprojection
  * Images provided are already rectified thus Q matrix can be composed from calibration data Via **`Data Set`** 


<hr>

### Virtual ENV Dependencies for WSL

1. ```
   python3 -m venv open3d_env
   ```
2. ```
   source open3d_env/bin/activate
   ```
3. ```
   pip install open3d numpy opencv-python
   ```

[**`Data Set`**](https://vision.middlebury.edu/stereo/data/scenes2014/)
