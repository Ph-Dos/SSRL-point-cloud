# Functionality

#### 1️⃣ Disparity Map Generation
* Disparity map is generated with Simple Stero
  * Disparity values are computed using triangulation of corrisponding points Determined by pixel positions from Left and Right Images
  * Corrisponding points are determined by locating the most similiar pixel pattern using a kernel scanned over an Epipolar line
      * Only works if y-values in Left and Right images are Identical


#### 2️⃣ Point Cloud Generation
* Point cloud is generated using 3D rectification
  * Images provided are already rectified thus Q matrix can be composed from calibration data Via **`Data Set`** 


<hr>

### Virtual ENV Dependencies for WSL
* `python3 -m venv open3d_env source open3d_env/bin/activate`
* `pip install open3d numpy opencv-python`

[**`Data Set`**](https://vision.middlebury.edu/stereo/data/scenes2014/)
