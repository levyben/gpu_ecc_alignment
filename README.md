# CUDA ECC image alignment algorithm

This is my implementation of the ECC image alignment algorithm for GPU, using CUDA and NPP (NVIDIA 2D Image And Signal Performance Primitives) libraies. 

**As far as I concern, this is the only optimization available over OpenCV's CPU implemenation.**

The ECC Finds the geometric transform (warp) between two images in terms of the ECC criterion.

# Paper:
```
Georgios D Evangelidis and Emmanouil Z Psarakis. Parametric image alignment using enhanced correlation coefficient maximization.
Pattern Analysis and Machine Intelligence, IEEE Transactions on, 30(10):1858–1865, 2008.
```
# Limitations: 
1) Supports only 1-Channel images (grayscale). Implementation for 3-Channel is trivial. 
2) CMakeLists file might not suffice for linking library dependencies (OpenCV and NPP) and folder includes, add it by yourself.

# Requirements:
1) Opencv compiled with CUDA 10.2
2) CUDA Toolkit (has the NVIDIA NPP libaries) 

# Benchmark

Benchmark was conducted using the following command argument (the input image is obtained automatically from opencv installation directory):
```
./ecc_gpu fruits.jpg -o=outWarp.ecc -m=homography -e=1e-6 -N=70 -v=1
```
On **1920x2560** (rowsxcols) image resolution:

| Device        | Time [sec]           | Implementation |
| ------------- |:-------------:| ------------- |
| Single-Thread CPU (Intel i7-6700)      | 20.865 | OpenCV CPU (Using AVX2)
| GPU (Nvidia GTX 970 - NO OC)      | 2.187      |  OpenCV GPU+NPP

# Conclusions

Order of magnitude (x10~) speed-up !!

Expect better speed-up with better GPU and harder alignment cases (bad alignment initilization, more iterations, small epsilon).
