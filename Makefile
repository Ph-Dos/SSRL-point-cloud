CC = python
APP = Disparity-to-PointCloud.py
DEP = Motorcycle/im0.png Motorcycle/im1.png Motorcycle/calib.txt

ply:
	$(CC) $(APP) $(DEP)
