#pragma once
#include "opencv2/opencv.hpp"

using namespace cv;

double findTransformECCGpu(InputArray templateImage, InputArray inputImage,
	InputOutputArray warpMatrix, int motionType,
	TermCriteria criteria, int gaussFiltSize,
	InputArray inputMask= cv::noArray());