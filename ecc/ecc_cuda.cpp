#include "ecc_cuda.h"
#include "opencv2/video.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core.hpp"
#include "cuda.h"
#include <cuda_runtime_api.h>
#include "npp.h"
#include "npps.h"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudabgsegm.hpp"
#include <opencv2/cudaarithm.hpp>
/****************************************************************************************\
*                                      Cuda Image Alignment (ECC algorithm)                  *
\****************************************************************************************/

using namespace cv;


struct ECC_GPU_Buffers
{

	cuda::GpuMat den_, hatX_, hatY_, src1Divided_, src2Divided_, temp_, dst;
	cuda::GpuMat src1, src2, src3, src4, src5;
	cuda::GpuMat hatysrc2, hatxsrc1;
	cuda::GpuMat imageWarped, imageProjection, gradientX, gradientY, imageMask, imageFloat,preMask, templateFloat, templateZM, error;
	cuda::Stream stream;
	double *pDp_dev;
	double *mean_dev, *stddev_dev;
	Npp8u *pDeviceBuffer;
	Npp64f *pNorm;
	Npp8u* normL2GPUFloatMatBuf;
	Npp8u* meanStdDevBuf;
	ECC_GPU_Buffers(const cv::Size& size, int params)
	{
		int h = size.height;
		int w = size.width;
		hatxsrc1.create(size, CV_32F);
		hatysrc2.create(size, CV_32F);
		dst.create(cv::Size(params*w,h), CV_32F);
		den_.create(size,CV_32F);
		hatX_.create(size, CV_32F);
		hatY_.create(size, CV_32F);
		src1Divided_.create(size, CV_32F);
		src2Divided_.create(size, CV_32F);
		temp_.create(size, CV_32F);

		src1.create(size, CV_32F);
		src2.create(size, CV_32F);
		src3.create(size, CV_32F);
		src4.create(size, CV_32F);
		src5.create(size, CV_32F);


		imageWarped.create(size, CV_32F);
		imageProjection.create(size, CV_32F);
		gradientX.create(size, CV_32F);
		gradientY.create(size, CV_32F);
		imageMask.create(size, CV_8U);
		imageFloat.create(size, CV_32F);
		preMask.create(size, CV_8U);
		templateFloat.create(size, CV_32F);
		templateZM.create(size, CV_32F);
		error.create(size, CV_32F);


		cudaMalloc((void**)&pDp_dev, sizeof(double)*params*params);
		cudaMalloc((void**)&mean_dev, sizeof(double));
		cudaMalloc((void**)&stddev_dev, sizeof(double));
		int hpBufferSize;
		NppiSize ns;
		ns.height = h;
		ns.width = w;

		nppiDotProdGetBufferHostSize_32f64f_C1R(ns, &hpBufferSize);
		cudaMalloc((void**)&pDeviceBuffer, sizeof(Npp8u)*hpBufferSize);
		cudaMalloc((void**)&pNorm, sizeof(Npp64f)*params);

		nppiNormL2GetBufferHostSize_32f_C1R(ns, &hpBufferSize);
		cudaMalloc((void**)&normL2GPUFloatMatBuf, sizeof(Npp8u)*hpBufferSize);

		nppiMeanStdDevGetBufferHostSize_32f_C1R(ns, &hpBufferSize);
		cudaMalloc((void**)&meanStdDevBuf, sizeof(Npp8u)*hpBufferSize);


	}
	~ECC_GPU_Buffers()
	{
		cudaFree(pDp_dev);
		cudaFree(mean_dev);
		cudaFree(stddev_dev);
		cudaFree(pDeviceBuffer);
		cudaFree(pNorm);
		cudaFree(normL2GPUFloatMatBuf);
		cudaFree(meanStdDevBuf);
	}
};


void normL2GPUFloatMat(cv::cuda::GpuMat src,int i, ECC_GPU_Buffers& buffers)
{
	NppiSize sz;
	sz.width = src.cols;
	sz.height = src.rows;
	nppiNorm_L2_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), sz, &buffers.pNorm[i], buffers.normL2GPUFloatMatBuf);
}

void meanStdDev_32FC1M(cv::cuda::GpuMat src, cv::cuda::GpuMat mask, double *mean, double *stddev, ECC_GPU_Buffers& buffers)
{
	//CV_Assert(src.type() == CV_32FC1);
	NppiSize sz;
	sz.width = src.cols;
	sz.height = src.rows;
	nppiMean_StdDev_32f_C1MR(src.ptr<Npp32f>(), static_cast<int>(src.step), mask.ptr<Npp8u>(), static_cast<int>(mask.step), sz, buffers.meanStdDevBuf, buffers.mean_dev, buffers.stddev_dev);

	cudaMemcpy(mean, buffers.mean_dev, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(stddev, buffers.stddev_dev, sizeof(double), cudaMemcpyDeviceToHost);
}

void dotGpuMat(cv::cuda::GpuMat m1, cv::cuda::GpuMat m2, int i, ECC_GPU_Buffers& buffers)
{
	NppiSize ns;
	double *pDp_dev = &buffers.pDp_dev[i];
	ns.height = m1.rows;
	ns.width = m1.cols;
	nppiDotProd_32f64f_C1R(m1.ptr<Npp32f>(), static_cast<int>(m1.step), m2.ptr<Npp32f>(), static_cast<int>(m2.step), ns, pDp_dev, buffers.pDeviceBuffer);
}


static void image_jacobian_homo_ECC_cuda(const Mat& src5, ECC_GPU_Buffers& gpuEccBuffers)
{
	const float* hptr = src5.ptr<float>(0);

	const float h0_ = hptr[0];
	const float h1_ = hptr[3];
	const float h2_ = hptr[6];
	const float h3_ = hptr[1];
	const float h4_ = hptr[4];
	const float h5_ = hptr[7];
	const float h6_ = hptr[2];
	const float h7_ = hptr[5];

	const int w = gpuEccBuffers.src1.cols;

	cuda::addWeighted(gpuEccBuffers.src3, h2_, gpuEccBuffers.src4, h5_, 1.0, gpuEccBuffers.den_,CV_32F, gpuEccBuffers.stream);
	
	//create projected points
	cuda::addWeighted(gpuEccBuffers.src3, -h0_, gpuEccBuffers.src4, -h3_, -h6_, gpuEccBuffers.hatX_, CV_32F, gpuEccBuffers.stream);

	cuda::divide(gpuEccBuffers.hatX_, gpuEccBuffers.den_, gpuEccBuffers.hatX_,1.0,CV_32F, gpuEccBuffers.stream);

	cuda::addWeighted(gpuEccBuffers.src3, -h1_, gpuEccBuffers.src4, -h4_, -h7_, gpuEccBuffers.hatY_,CV_32F, gpuEccBuffers.stream);

	cuda::divide(gpuEccBuffers.hatY_, gpuEccBuffers.den_, gpuEccBuffers.hatY_,1.0,CV_32F, gpuEccBuffers.stream);


	//instead of dividing each block with den,
	//just pre-divide the block of gradients (it's more efficient)
	cuda::divide(gpuEccBuffers.src1, gpuEccBuffers.den_, gpuEccBuffers.dst.colRange(6 * w, 7 * w), 1.0,CV_32F, gpuEccBuffers.stream);
	cuda::divide(gpuEccBuffers.src2, gpuEccBuffers.den_, gpuEccBuffers.dst.colRange(7 * w, 8 * w), 1.0, CV_32F, gpuEccBuffers.stream);

	

	//compute Jacobian blocks (8 blocks)

	cuda::GpuMat& src1Divided_ = gpuEccBuffers.dst.colRange(6 * w, 7 * w);
	cuda::GpuMat& src2Divided_ = gpuEccBuffers.dst.colRange(7 * w, 8 * w);
	
	cuda::multiply(src1Divided_, gpuEccBuffers.src3, gpuEccBuffers.dst.colRange(0, w), 1.0, CV_32F, gpuEccBuffers.stream);

	cuda::multiply(src2Divided_, gpuEccBuffers.src3, gpuEccBuffers.dst.colRange(w, 2 * w), 1.0, CV_32F, gpuEccBuffers.stream);

	
	cuda::multiply(gpuEccBuffers.hatX_, src1Divided_, gpuEccBuffers.hatxsrc1, 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::multiply(gpuEccBuffers.hatY_, src2Divided_, gpuEccBuffers.hatysrc2, 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::add(gpuEccBuffers.hatysrc2, gpuEccBuffers.hatxsrc1, gpuEccBuffers.temp_,noArray(), CV_32F, gpuEccBuffers.stream);

	cuda::multiply(gpuEccBuffers.temp_, gpuEccBuffers.src3, gpuEccBuffers.dst.colRange(2 * w, 3 * w), 1.0, CV_32F, gpuEccBuffers.stream);
	
	cuda::multiply(src1Divided_, gpuEccBuffers.src4, gpuEccBuffers.dst.colRange(3 * w, 4 * w), 1.0, CV_32F, gpuEccBuffers.stream);

	cuda::multiply(src2Divided_, gpuEccBuffers.src4, gpuEccBuffers.dst.colRange(4 * w, 5 * w), 1.0, CV_32F, gpuEccBuffers.stream);

	cuda::multiply(gpuEccBuffers.temp_, gpuEccBuffers.src4, gpuEccBuffers.dst.colRange(5 * w, 6 * w), 1.0, CV_32F, gpuEccBuffers.stream);
}


static void image_jacobian_euclidean_ECC_cuda(const Mat& src5, ECC_GPU_Buffers& gpuEccBuffers)
{

	const float* hptr = src5.ptr<float>(0);

	const float h0 = hptr[0];//cos(theta)
	const float h1 = hptr[3];//sin(theta)

	const int w = gpuEccBuffers.src1.cols;


	//create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
	cuda::addWeighted(gpuEccBuffers.src3, -h1, gpuEccBuffers.src4, -h0, 0.0, gpuEccBuffers.hatX_, CV_32F, gpuEccBuffers.stream);

	//create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
	cuda::addWeighted(gpuEccBuffers.src3, h0, gpuEccBuffers.src4, -h1, 0.0, gpuEccBuffers.hatY_, CV_32F, gpuEccBuffers.stream);


	//compute Jacobian blocks (3 blocks)
	cuda::multiply(gpuEccBuffers.src1, gpuEccBuffers.hatX_, gpuEccBuffers.hatxsrc1, 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::multiply(gpuEccBuffers.src2, gpuEccBuffers.hatY_, gpuEccBuffers.hatysrc2, 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::addWeighted(gpuEccBuffers.hatxsrc1, 1.0, gpuEccBuffers.hatysrc2, 1.0, 0.0, gpuEccBuffers.dst.colRange(0, w), CV_32F, gpuEccBuffers.stream);

	gpuEccBuffers.src1.copyTo(gpuEccBuffers.dst.colRange(w, 2 * w));
	gpuEccBuffers.src2.copyTo(gpuEccBuffers.dst.colRange(2 * w, 3 * w));
}


static void image_jacobian_affine_ECC_cuda(ECC_GPU_Buffers& gpuEccBuffers)
{
	CV_Assert(gpuEccBuffers.dst.cols == (6 * gpuEccBuffers.src1.cols));

	const int w = gpuEccBuffers.src1.cols;

	//compute Jacobian blocks (6 blocks)
	cuda::multiply(gpuEccBuffers.src1, gpuEccBuffers.src3, gpuEccBuffers.dst.colRange(0 , w), 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::multiply(gpuEccBuffers.src2, gpuEccBuffers.src3, gpuEccBuffers.dst.colRange(w, 2 * w), 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::multiply(gpuEccBuffers.src1, gpuEccBuffers.src4, gpuEccBuffers.dst.colRange(2 * w, 3 * w), 1.0, CV_32F, gpuEccBuffers.stream);
	cuda::multiply(gpuEccBuffers.src2, gpuEccBuffers.src4, gpuEccBuffers.dst.colRange(3 * w, 4 * w), 1.0, CV_32F, gpuEccBuffers.stream);
	gpuEccBuffers.src1.copyTo(gpuEccBuffers.dst.colRange(4 * w, 5 * w));
	gpuEccBuffers.src2.copyTo(gpuEccBuffers.dst.colRange(5 * w, 6 * w));
}

static void image_jacobian_translation_ECC_cuda(ECC_GPU_Buffers& gpuEccBuffers)
{
	const int w = gpuEccBuffers.src1.cols;
	//compute Jacobian blocks (2 blocks)
	gpuEccBuffers.src1.copyTo(gpuEccBuffers.dst.colRange(0, w));
	gpuEccBuffers.src2.copyTo(gpuEccBuffers.dst.colRange(w, 2 * w));

}

static void project_onto_jacobian_ECC_cuda(const cuda::GpuMat& src1, const cuda::GpuMat& src2, Mat& dst, ECC_GPU_Buffers& eccBuffers)
{
	/* this functions is used for two types of projections. If src1.cols ==src.cols
	it does a blockwise multiplication (like in the outer product of vectors)
	of the blocks in matrices src1 and src2 and dst
	has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
	(number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

	The number_of_blocks is equal to the number of parameters we are lloking for
	(i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

	*/
	CV_Assert(src1.rows == src2.rows);
	CV_Assert((src1.cols % src2.cols) == 0);
	int w;

	float* dstPtr = dst.ptr<float>(0);

	if (src1.cols != src2.cols) {//dst.cols==1
		w = src2.cols;
		std::vector<double> dotProdDoubles(dst.rows);
		for (int i = 0; i < dst.rows; i++) {
			dotGpuMat(src2, src1.colRange(i*w, (i + 1)*w),i, eccBuffers);
		}
		cudaMemcpy(dotProdDoubles.data(), eccBuffers.pDp_dev, sizeof(double)*dst.rows, cudaMemcpyDeviceToHost);
		std::copy(dotProdDoubles.begin(), dotProdDoubles.end(), dstPtr);
	}

	else {
		CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
		w = src2.cols / dst.cols;
		std::vector<double> dotProdDoubles(dst.rows*dst.cols);
		std::vector<double> normDoubles(dst.rows);
		
		for (int i = 0; i < dst.rows; i++) {
			normL2GPUFloatMat(src1.colRange(i*w, (i + 1)*w),i, eccBuffers);
			for (int j = i + 1; j < dst.cols; j++) { //j starts from i+1
				dotGpuMat(src1.colRange(i*w, (i + 1)*w), src2.colRange(j*w, (j + 1)*w), i*dst.cols + j, eccBuffers);
			}
		}
		cudaMemcpy(dotProdDoubles.data(), eccBuffers.pDp_dev, sizeof(double)*dst.rows*dst.cols, cudaMemcpyDeviceToHost);
		cudaMemcpy(normDoubles.data(), eccBuffers.pNorm, sizeof(double)*dst.rows, cudaMemcpyDeviceToHost);

		for (int i = 0; i < dst.rows; i++) {
			dstPtr[i*(dst.rows + 1)] = normDoubles[i] * normDoubles[i]; //diagonal elements
			for (int j = i + 1; j < dst.cols; j++) { //j starts from i+1
				dstPtr[i*dst.cols + j] = float(dotProdDoubles[i*dst.cols + j]);
				dstPtr[j*dst.cols + i] = dstPtr[i*dst.cols + j]; //due to symmetry

			}
		}

	}
}

static void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType)
{
    CV_Assert (map_matrix.type() == CV_32FC1);
    CV_Assert (update.type() == CV_32FC1);

    CV_Assert (motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
        motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert (map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert (map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert (map_matrix.rows == 2 && update.rows == 2);

    CV_Assert (update.cols == 1);

    CV_Assert( map_matrix.isContinuous());
    CV_Assert( update.isContinuous() );


    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);


    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float) cos(new_theta);
        mapPtr[3] = (float) sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}


double findTransformECCGpu_(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria,
                            InputArray inputMask,
                            int gaussFiltSize)
{


    Mat src = templateImage.getMat();//template image
    Mat dst = inputImage.getMat(); //input image (to be warped)
    Mat map = warpMatrix.getMat(); //warp (transformation)


    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());

    // If the user passed an un-initialized warpMatrix, initialize to identity
    if(map.empty()) {
        int rowCount = 2;
        if(motionType == MOTION_HOMOGRAPHY)
            rowCount = 3;

        warpMatrix.create(rowCount, 3, CV_32FC1);
        map = warpMatrix.getMat();
        map = Mat::eye(rowCount, 3, CV_32F);
    }

    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);

    CV_Assert (motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
        motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY){
        CV_Assert (map.rows ==3);
    }

    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps    = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  :  -1;

    int paramTemp = 8;
    switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
	  case MOTION_AFFINE:
		  paramTemp = 6;
		  break;
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
    }

	auto gpubuffers = ECC_GPU_Buffers(cv::Size(src.cols, src.rows), paramTemp);

    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;

    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

    Xcoord.release();
    Ycoord.release();

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat imageMask     = Mat(hs, ws, CV_8U); // to store the final mask

    Mat inputMaskMat = inputMask.getMat();
    //to use it for mask warping
    Mat preMask;
    if(inputMask.empty())
        preMask = Mat::ones(hd, wd, CV_8U);
    else
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);

    //gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    Mat preMaskFloat;
    preMask.convertTo(preMaskFloat, CV_32F);
    GaussianBlur(preMaskFloat, preMaskFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
    // Change threshold.
    preMaskFloat *= (0.5/0.95);
    // Rounding conversion.
    preMaskFloat.convertTo(preMask, preMask.type());
    preMask.convertTo(preMaskFloat, preMaskFloat.type());

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
    Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());

    gradientX = gradientX.mul(preMaskFloat);
    gradientY = gradientY.mul(preMaskFloat);

    // matrices needed for solving linear equation system for maximizing ECC
    Mat jacobian                = Mat(hs, ws*numberOfParameters, CV_32F);
    Mat hessian                 = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv              = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection         = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;


	gpubuffers.src3.upload(Xgrid);
	gpubuffers.src4.upload(Ygrid);
	gpubuffers.gradientX.upload(gradientX);
	gpubuffers.gradientY.upload(gradientY);
	gpubuffers.imageFloat.upload(imageFloat);
	gpubuffers.imageMask.upload(imageMask);
	gpubuffers.preMask.upload(preMask);
	gpubuffers.templateFloat.upload(templateFloat);

    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {

        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY)
        {
			cuda::warpAffine(gpubuffers.imageFloat, gpubuffers.imageWarped, map, gpubuffers.imageWarped.size(), imageFlags);
			cuda::warpAffine(gpubuffers.gradientX, gpubuffers.src1, map, gpubuffers.src1.size(), imageFlags);
			cuda::warpAffine(gpubuffers.gradientY, gpubuffers.src2, map, gpubuffers.src2.size(), imageFlags);
			cuda::warpAffine(gpubuffers.preMask, gpubuffers.imageMask, map, gpubuffers.imageMask.size(), maskFlags);

        }
        else
        {
			cuda::warpPerspective(gpubuffers.imageFloat, gpubuffers.imageWarped, map, gpubuffers.imageWarped.size(), imageFlags);
			cuda::warpPerspective(gpubuffers.gradientX, gpubuffers.src1, map, gpubuffers.src1.size(), imageFlags);
			cuda::warpPerspective(gpubuffers.gradientY, gpubuffers.src2, map, gpubuffers.src2.size(), imageFlags);
			cuda::warpPerspective(gpubuffers.preMask, gpubuffers.imageMask, map, gpubuffers.imageMask.size(), maskFlags);
			
        }
		
        double imgMean, imgStd;
		double tmpMean, tmpStd;
		meanStdDev_32FC1M(gpubuffers.templateFloat, gpubuffers.imageMask, &tmpMean, &tmpStd, gpubuffers);
		meanStdDev_32FC1M(gpubuffers.imageWarped, gpubuffers.imageMask, &imgMean, &imgStd, gpubuffers);
        cuda::subtract(gpubuffers.imageWarped,   imgMean, gpubuffers.imageWarped, gpubuffers.imageMask);//zero-mean input
        cuda::subtract(gpubuffers.templateFloat, tmpMean, gpubuffers.templateZM, gpubuffers.imageMask);//zero-mean template
		
        const double tmpNorm = std::sqrt(cuda::countNonZero(gpubuffers.imageMask)*(tmpStd)*(tmpStd));
        const double imgNorm = std::sqrt(cuda::countNonZero(gpubuffers.imageMask)*(imgStd)*(imgStd));

        // calculate jacobian of image wrt parameters
        switch (motionType){
            case MOTION_AFFINE:
                image_jacobian_affine_ECC_cuda(gpubuffers);
                break;
            case MOTION_HOMOGRAPHY:
				image_jacobian_homo_ECC_cuda(map, gpubuffers);
                break;
            case MOTION_TRANSLATION:
                image_jacobian_translation_ECC_cuda(gpubuffers);
                break;
            case MOTION_EUCLIDEAN:
                image_jacobian_euclidean_ECC_cuda(map, gpubuffers);
                break;
        }
		cuda::GpuMat& jacobianGPU = gpubuffers.dst;
        // calculate Hessian and its inverse
		gpubuffers.stream.waitForCompletion();
		project_onto_jacobian_ECC_cuda(jacobianGPU, jacobianGPU, hessian, gpubuffers);
        hessianInv = hessian.inv();
		double correlation;
		dotGpuMat(gpubuffers.templateZM, gpubuffers.imageWarped,0, gpubuffers); //templateZM.dot(imageWarped);
		cudaMemcpy(&correlation, gpubuffers.pDp_dev, sizeof(double), cudaMemcpyDeviceToHost);
		 
        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation/(imgNorm*tmpNorm);
        if (cvIsNaN(rho)) {
          CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // project images into jacobian
        project_onto_jacobian_ECC_cuda( jacobianGPU, gpubuffers.imageWarped, imageProjection, gpubuffers);
        project_onto_jacobian_ECC_cuda(jacobianGPU, gpubuffers.templateZM, templateProjection, gpubuffers);


        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0)
        {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

        }
        const double lambda = (lambda_n/lambda_d);

        // estimate the update step delta_p
		cuda::addWeighted(gpubuffers.templateZM, lambda, gpubuffers.imageWarped, -1.0, 0.0, gpubuffers.error, CV_32F);
        project_onto_jacobian_ECC_cuda(jacobianGPU, gpubuffers.error, errorProjection, gpubuffers);
        deltaP = hessianInv * errorProjection;

        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);


    }

    // return final correlation coefficient
    return rho;
}

double findTransformECCGpu(InputArray templateImage, InputArray inputImage,
    InputOutputArray warpMatrix, int motionType,
    TermCriteria criteria, int gaussianFilterSize,
    InputArray inputMask)
{
    return findTransformECCGpu_(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussianFilterSize);
}

/* End of file. */
