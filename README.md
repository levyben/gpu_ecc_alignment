# CUDA ECC image alignement algorithm

This is my implementation of the ECC image alignment algorithm for GPU, using CUDA and NPP (NVIDIA 2D Image And Signal Performance Primitives) libraies. 

The ECC Finds the geometric transform (warp) between two images in terms of the ECC criterion.

Paper:
```
Georgios D Evangelidis and Emmanouil Z Psarakis. Parametric image alignment using enhanced correlation coefficient maximization.
Pattern Analysis and Machine Intelligence, IEEE Transactions on, 30(10):1858–1865, 2008.
```
Limitation: 
Supports only 1-Channel images (grayscale). Implementation for 3-Channel is trivial. 

Benchmark was conducted using the following command argument:
```
./ecc_gpu fruits.jpg -o=outWarp.ecc -m=homography -e=1e-6 -N=70 -v=1
```
On **1920x2560** (rowsxcols) image resolution:

| Device        | Time [sec]           | 
| ------------- |:-------------:|
| Single-Thread CPU (Intel i7-6700)      | 20.865 | 
| GPU (Nvidia GTX 970 - NO OC)      | 2.187      |  


Order of magnitude (x10~) speed-up !!

Expect better speed-up with better GPU and harder alignment cases (bad alignment initilization, more iterations, small epsilon).
