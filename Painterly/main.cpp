#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <random> 
#include <time.h>	
#include <iostream>
#include <vector>

#include "header.h"

using namespace cv;
using namespace std;

#define SHOW(a) std::cout << #a << "= " << endl << (a) << std::endl
#define PI 3.1415926535
#define TWOPI 2*PI

mt19937 mt;
// Steps:
//  - seperate into high + low frequencies
//  - sample high, low
//  - 
int main(int argc, char* argv[]) {
	mt.seed( 1234 );

	Mat brush;
	if (argc > 1)
		brush = imread(argv[1]);  
	else
		brush = imread("longBrush2.png");

       if (argc > 2 && std::string(argv[2]) == "--novid") {
              Paint(brush, 30, true);
       } else {
              Paint(brush, 30);
       }

	return 0;
}

void Paint(Mat brush, int size, bool novid, int N, int nAngles, float noise) {
	brush.convertTo(brush, CV_32F, 1/255.);

	if (brush.channels() == 3)
		cvtColor(brush, brush, CV_BGR2GRAY);

	int maxDim = max(brush.cols, brush.rows);
	float k = size/(float)maxDim;
	resize(brush, brush, Size(0,0), k, k);

	Mat brush_small;
	resize(brush, brush_small, Size(0,0), 0.25, 0.25);

	vector<Mat> brushes, brushes_small;
	computeRotations(brush, brushes);
	computeRotations(brush_small, brushes_small);

	Mat importance;

	Mat orientation;
	vector<float> tensor; 
    

       VideoCapture cap(0);
       if (!novid) {
              if (!cap.isOpened()) {
                     printf("video not open\n");
                     return;
              }
       }

	namedWindow("vid", WINDOW_NORMAL);
	Mat frame;
	Mat out;
	while (true)
	{
              if (novid) {
                     frame = imread("chase.jpg");
              } else {
                     cap >> frame;
              }
		frame.convertTo(frame, CV_32F, 1/255.);

		orientation = Mat::zeros(frame.rows, frame.cols, CV_32F);
		out = Mat::zeros(frame.size(), CV_32FC3);

		computerTensor(frame, tensor);
		computeOrientation(tensor, orientation);
		singleScaleOrientedPaint(frame, out, orientation, Mat::ones(frame.rows, frame.cols, CV_32F), brushes, 10, 10000);

		sharpnessMap(frame, importance, 2);
		singleScaleOrientedPaint(frame, out, orientation, importance, brushes_small, 10, 5000);

		imshow("vid", out);
		if(waitKey(1) >= 0) break;
	}

	return;
}

// im must have 3 channels
bool applyStroke(Mat& im, int y, int x, Vec3f rgb, const Mat& brush) {
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

void singleScalePaint(const Mat& im, Mat& out, const Mat& importance, const Mat& brush, int size, int N, float noise) {
	mt.seed( 1234 );

	uniform_int_distribution<int> randX(0, im.cols-1);
	uniform_int_distribution<int> randY(0, im.rows-1);
	uniform_real_distribution<float> rand(0,1);
	normal_distribution<float> randNrm(0, noise);

	N /= max(1e-1,cv::mean(importance)[0]);
	int x,y;
	float r;
	for (int i = 0; i < N; i++) {
		x = randX(mt);
		y = randY(mt);
		r = rand(mt);

		if (r > importance.at<float>(y,x))
			continue;

		applyStroke(out, y, x, im.at<Vec3f>(y,x), brush);
	}
}


void singleScaleOrientedPaint(const Mat& im, Mat& out, const Mat& orientaiton, const Mat& importance, 
							  const vector<Mat>& brushes, int size, int N, float noise) 
{

	uniform_int_distribution<int> randX(0, im.cols-1);
	uniform_int_distribution<int> randY(0, im.rows-1);
	uniform_real_distribution<float> rand(0,1);
	normal_distribution<float> randNrm(0, noise);

	N /= max(1e-1,cv::mean(importance)[0]);
	int x,y;
	float r;
	for (int i = 0; i < N; i++) {
		x = randX(mt);
		y = randY(mt);
		r = rand(mt);

		if (r > importance.at<float>(y,x))
			continue;

		int index = orientaiton.at<float>(y,x)*12;
		if (index == 12)
			index--;
		applyStroke(out, y, x, im.at<Vec3f>(y,x), brushes[index]);
	}
}

int sharpnessMap(const Mat& im, Mat& out, float sigma) {
	Mat im_grey;

	if (im.channels() == 1)
		im_grey = im.clone();
	else 
		cvtColor(im, im_grey, CV_BGR2GRAY);

	if (out.empty()) {
		out.create(im_grey.size(), im_grey.type());
	}

	Size ksize = Size(9 + 4*((int)sigma-1), 9 + 4*((int)sigma-1));
	Size ksize2 = Size(9 + 4*((int)(4*sigma)-1), 9 + 4*((int)(4*sigma)-1));

	GaussianBlur(im_grey, out, ksize, sigma, sigma);

	out = im_grey - out;
	out = out.mul(out);

	GaussianBlur(out, out, ksize2, 4*sigma, 4*sigma);
	normalize(out);

	return 1;
}

void normalize(Mat& im) {
	double min, max;
	minMaxIdx(im, &min, &max, 0, 0);

	if ( (max - min) < 1e-4 )
		return; 

	im = (im - min) / (max - min);
}

void computerTensor(const cv::Mat& im, vector<float>& tensor, float sigma, float factor) {
	int h = im.rows;
	int w = im.cols;
	int N = h*w; 
	tensor.resize(N*3);

	Size ksize = Size(9 + 4*((int)sigma-1), 9 + 4*((int)sigma-1));
	Mat im_g;

	if (im.channels() == 1)
		im_g = im.clone();
	else 
		cvtColor(im, im_g, CV_BGR2GRAY);

	cv::pow(im_g, 0.5, im_g);
	GaussianBlur(im_g, im_g, ksize, sigma, sigma);

	Mat dx, dy;

	Sobel(im_g, dx, -1, 1, 0);
	Sobel(im_g, dy, -1, 0, 1);

	Mat dxx, dyx, dyy;

	dxx = dx.mul(dx);
	dyx = dx.mul(dy);
	dyy = dy.mul(dy);


	GaussianBlur(dxx, dxx, ksize, factor*sigma, factor*sigma);
	GaussianBlur(dyx, dyx, ksize, factor*sigma, factor*sigma);
	GaussianBlur(dyy, dyy, ksize, factor*sigma, factor*sigma);

	int index = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tensor[index++] = dyy.at<float>(i,j);
			tensor[index++] = dyx.at<float>(i,j);
			tensor[index++] = dxx.at<float>(i,j);
		}
	}
}

void computeOrientation(const vector<float>& tensor, Mat& orientation) {
	vector<float> eig(2*tensor.size());

	const float* tensor_ptr = tensor.data();
	float* eig_ptr = eig.data();

	int N = tensor.size() / 3;

	eigen2x2(tensor_ptr, eig_ptr, N);

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

void computeRotations(const Mat& brush, vector<Mat>& rotations, int nAngles) {
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
	float dTheta = 180. / nAngles;
	rotations.resize(nAngles);
	for (int i = 0; i < nAngles; i++) {
		rotations[i].create(newHeight, newWidth, brush.type());
		Mat R = getRotationMatrix2D(pc, theta, 1);
		warpAffine(centered, rotations[i], R, rotations[i].size());
		theta += dTheta;
	}
}
