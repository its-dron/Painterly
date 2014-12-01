#include "Painter.h"

#include <opencv2\imgproc\imgproc.hpp>

#include <random>
#include <time.h>

using namespace std;
using namespace cv;

//TODO more documentation

#define PI 3.1415926535

Painter::Painter(void)
{
	m_mt.seed(time(NULL));
	srand (time(NULL));
}

Painter::Painter(const Mat& brush, int numAngles, int brushSize)
{
	Painter();

	m_originalBrush = brush.clone();
	m_numAngles = numAngles;
	m_strokeSize = (brushSize > 0 ? brushSize : max(brush.cols, brush.rows));

	updateBrush();
}


/// Paints the specified frame.
/// frame: image frame to create painting of.
/// out: Mat to write to.
void Painter::Paint(Mat frame, Mat& out) {
	frame.convertTo(frame, CV_32F, 1/255.);

	Mat orientation = Mat::zeros(frame.rows, frame.cols, CV_32F);
	out = Mat::zeros(frame.size(), CV_32FC3);

	computeOrientation(frame, orientation);
	singleScaleOrientedPaint(frame, out, orientation, Mat::ones(frame.rows, frame.cols, CV_32F), m_brushes, 10000);

	Mat details;
	sharpnessMap(frame, details, 2);
	singleScaleOrientedPaint(frame, out, orientation, details, m_brushesSmall, 5000);
}


/// Paints the large scale details.
/// frame: The frame.
/// out: Mat to write to.
/// oriented: Defaults to true. Set to false to disable oriented painting
void Painter::PaintLargeScale(Mat frame, Mat& out, bool oriented) {
	frame.convertTo(frame, CV_32F, 1/255.);

	out = Mat::zeros(frame.size(), CV_32FC3);

	if (oriented) {
		Mat orientation = Mat::zeros(frame.rows, frame.cols, CV_32F);
		computeOrientation(frame, orientation);
		singleScaleOrientedPaint(frame, out, orientation, Mat::ones(frame.rows, frame.cols, CV_32F), m_brushes, 10000);
	} else {
		singleScalePaint(frame, out, Mat::ones(frame.rows, frame.cols, CV_32F), m_brushes[0], 10000);
	}
}


/// Paints the small scale.
/// frame: The frame.
/// out: Mat to write to.
/// oriented: Defaults to true. Set to false to disable oriented painting
void Painter::PaintSmallScale(Mat frame, Mat& out, bool oriented) {
	frame.convertTo(frame, CV_32F, 1/255.);

	out = Mat::zeros(frame.size(), CV_32FC3);

	Mat details;
	sharpnessMap(frame, details, 2);

	if (oriented) {
		Mat orientation = Mat::zeros(frame.rows, frame.cols, CV_32F);
		computeOrientation(frame, orientation);
		singleScaleOrientedPaint(frame, out, orientation, details, m_brushesSmall, 5000);
	} else {
		singleScalePaint(frame, out, details, m_brushesSmall[0], 5000);
	}
}


/// Singles the scale paint.
/// im: The im.
/// out: The out.
/// importance: The importance.
/// brush: The brush.
/// N: The n.
/// noise: The noise.
void Painter::singleScalePaint(const Mat& im, Mat& out, const Mat& importance, 
							   const Mat& brush, int N, float noise)
{
	uniform_int_distribution<int> randX(0, im.cols-1);
	uniform_int_distribution<int> randY(0, im.rows-1);
	uniform_real_distribution<float> rand(0,1);
	normal_distribution<float> randNrm(0, noise);

	N /= max(1e-1,mean(importance)[0]);
	int x,y;
	float r;
	for (int i = 0; i < N; i++) {
		x = randX(m_mt);
		y = randY(m_mt);
		r = rand(m_mt);

		if (r > importance.at<float>(y,x))
			continue;

		applyStroke(out, y, x, im.at<Vec3f>(y,x), brush);
	}
}


/// Singles the scale oriented paint.
/// im: The im.
/// out: The out.
/// orientation: The orientation.
/// importance: The importance.
/// brushes: The brushes.
/// N: The n.
/// noise: The noise.
void Painter::singleScaleOrientedPaint(const Mat& im, Mat& out, const Mat& orientation, const Mat& importance, 
									   const vector<Mat>& brushes, int N, float noise)
{
	uniform_int_distribution<int> randX(0, im.cols-1);
	uniform_int_distribution<int> randY(0, im.rows-1);
	uniform_real_distribution<float> rand(0,1);
	normal_distribution<float> randNrm(0, noise);

	N /= max(1e-1,mean(importance)[0]);
	int x,y;
	float r;
	for (int i = 0; i < N; i++) {
		x = randX(m_mt);
		y = randY(m_mt);
		r = rand(m_mt);

		if (r > importance.at<float>(y,x))
			continue;

		int index = orientation.at<float>(y,x)*m_numAngles;
		if (index == m_numAngles)
			index = 0;
		applyStroke(out, y, x, im.at<Vec3f>(y,x), brushes[index]);
	}
}

/// Updates the brush.
void Painter::updateBrush() {
	m_brushes.resize(m_numAngles);
	m_brushesSmall.resize(m_numAngles);

	Mat brush = m_originalBrush.clone();
	Mat brush_small;

	int maxDim = max(brush.cols, brush.rows);
	float k = m_strokeSize/(float)maxDim;
	if (abs(1-k) > 1e-3)
		resize(brush, brush, Size(0,0), k, k);
	resize(brush, brush_small, Size(0,0), 0.25, 0.25);

	computeRotations(brush, m_brushes);
	computeRotations(brush_small, m_brushesSmall);
}


/// Computes the rotations.
/// brush: The brush.
/// rotations: The rotations.
void Painter::computeRotations(const Mat& brush, vector<Mat>& rotations) {
	int h = brush.rows;
	int w = brush.cols;

	int diagonal = (int)sqrt(w*w+h*h);

	int newWidth = diagonal;
	int newHeight = diagonal;
	Point2f pc(newHeight/2., newWidth/2.);

	int offsetX = (newWidth - w) / 2;
	int offsetY = (newHeight - h) / 2;
	Mat centered = Mat::zeros(newHeight, newWidth, brush.type());
	brush.copyTo(centered(Range(offsetY, offsetY+h), Range(offsetX, offsetX+w)));

	float theta = 0;
	float dTheta = 180. / m_numAngles;
	rotations.resize(m_numAngles);
	for (int i = 0; i < m_numAngles; i++) {
		rotations[i].create(newHeight, newWidth, brush.type());
		Mat R = getRotationMatrix2D(pc, theta, 1);
		warpAffine(centered, rotations[i], R, rotations[i].size());
		theta += dTheta;
	}
}


/// Computes the orientation.
/// frame: The frame.
/// orientation: The orientation.
void Painter::computeOrientation(const Mat& frame, Mat& orientation) {
	vector<float> tensor;
	computeTensor(frame, tensor);

	vector<float> eig(2*tensor.size());

	const float* tensor_ptr = tensor.data();
	float* eig_ptr = eig.data();

	int N = tensor.size() / 3;

	eigen2x2(tensor_ptr, eig_ptr, N);

	orientation.create(frame.rows, frame.cols, CV_32F);
	int w = orientation.cols;

	int offset = 6;
	int index = 0;
	float arg;
	for (int i = 0; i < orientation.rows; i++) {
		for (int j = 0; j < orientation.cols; j++) {
			index = offset*(i*w + j);
			arg = atan2f(eig_ptr[index+3], eig_ptr[index+2]); // eigen returns x,y
			arg /= PI;
			if (arg < 0)
				arg += 1;

			orientation.at<float>(i,j) = arg;
		}
	}
}


/// Computers the tensor.
/// im: The im.
/// tensor: The tensor.
/// sigma: The sigma.
/// factor: The factor.
void Painter::computeTensor(const Mat& im, vector<float>& tensor, float sigma, float factor) {
	int h = im.rows;
	int w = im.cols;
	int N = h*w; 
	tensor.resize(N*3);

	Size ksize = computeKernelSize(sigma);
	Size ksize2 = computeKernelSize(factor*sigma);

	Mat im_g;

	if (im.channels() == 1)
		im_g = im.clone();
	else 
		cvtColor(im, im_g, CV_BGR2GRAY);

	pow(im_g, 0.5, im_g);
	GaussianBlur(im_g, im_g, ksize, sigma, sigma);

	Mat dx, dy;

	Sobel(im_g, dx, -1, 1, 0);
	Sobel(im_g, dy, -1, 0, 1);

	Mat dxx, dyx, dyy;

	dxx = dx.mul(dx);
	dyx = dx.mul(dy);
	dyy = dy.mul(dy);


	GaussianBlur(dxx, dxx, ksize2, factor*sigma, factor*sigma);
	GaussianBlur(dyx, dyx, ksize2, factor*sigma, factor*sigma);
	GaussianBlur(dyy, dyy, ksize2, factor*sigma, factor*sigma);

	int index = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tensor[index++] = dyy.at<float>(i,j);
			tensor[index++] = dyx.at<float>(i,j);
			tensor[index++] = dxx.at<float>(i,j);
		}
	}
}


/// Sharpnesses the map.
/// im: The im.
/// out: The out.
/// sigma: The sigma.
int Painter::sharpnessMap(const Mat& im, Mat& out, float sigma) {
	Mat im_grey;

	if (im.channels() == 1)
		im_grey = im.clone();
	else 
		cvtColor(im, im_grey, CV_BGR2GRAY);

	if (out.empty()) {
		out.create(im_grey.size(), im_grey.type());
	}

	Size ksize = computeKernelSize(sigma);
	Size ksize2 = computeKernelSize(4*sigma);

	GaussianBlur(im_grey, out, ksize, sigma, sigma);

	out = im_grey - out;
	out = out.mul(out);

	GaussianBlur(out, out, ksize2, 4*sigma, 4*sigma);
	normalize(out);

	return 1;
}


/// Applies a single stroke.
/// im: The im.
/// y: The y.
/// x: The x.
/// rgb: The RGB.
/// brush: The brush.
/// <returns></returns>
bool Painter::applyStroke(Mat& im, int y, int x, Vec3f rgb, const Mat& brush) {
	if (im.channels() != 3 || brush.channels() != 1)
		return false;

	int w = brush.cols;
	int h = brush.rows;

	int w2 = w/2;
	int h2 = h/2;

	if (y < h2 || x < w2 || y >= im.rows-h+h2 || x >= im.cols-w+w2)
		return false;

	Mat stencil = im(Range(y-h2, y-h2+h), Range(x-w2, x-w2+w));

	float* stencil_ptr;
	const float* brush_ptr;
	for (int i = 0; i < h; i++) {
		stencil_ptr = stencil.ptr<float>(i);
		brush_ptr = brush.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			int j3 = j*3;
			stencil_ptr[j3+0] = stencil_ptr[j3+0]*(1-brush_ptr[j]) + brush_ptr[j]*rgb[0];
			stencil_ptr[j3+1] = stencil_ptr[j3+1]*(1-brush_ptr[j]) + brush_ptr[j]*rgb[1];
			stencil_ptr[j3+2] = stencil_ptr[j3+2]*(1-brush_ptr[j]) + brush_ptr[j]*rgb[2]; 
		}
	}

	return true;
}

/********************
* HELPER FUNCTIONS *
********************/


/// Normalizes the specified image to range [0,1]
/// im: image to normalize
void Painter::normalize(Mat& im) {
	double min, max;
	minMaxIdx(im, &min, &max, 0, 0);

	if ( (max - min) < 1e-4 )
		return; 

	im = (im - min) / (max - min);
}


/// Computes the size of the gaussain kernel based on the sigma.
/// This wast taken from scipy's gaussian blur.
Size Painter::computeKernelSize(float sigma) {
	return Size(9 + 4*((int)sigma-1), 9 + 4*((int)sigma-1));
}

