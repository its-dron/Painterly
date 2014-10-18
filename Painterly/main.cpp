#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <random> 
#include <time.h>	
#include <iostream>

#include "header.h"

using namespace cv;
using namespace std;

#define SHOW(a) std::cout << #a << "= " << endl << (a) << std::endl

mt19937 mt;
// Steps:
//  - seperate into high + low frequencies
//  - sample high, low
//  - 
int main(int argc, char* argv[]) {
	mt.seed( time(NULL) );

	if (argc < 3) 
		return -1;

	Mat im = imread(argv[1]);
	Mat brush = imread(argv[2]);

	im.convertTo(im, CV_32F,1/255.);
	brush.convertTo(brush, CV_32F, 1/255.);

	if (brush.channels() == 1)
		cvtColor(brush, brush, CV_GRAY2BGR); 

	//namedWindow("out");


	Mat importance(im.size(), im.type());
	sharpnessMap(im, importance);

	vector<float> tensor;
	computerTensor(im, tensor);
	computeOrientation(tensor, im);

	//while(1) {
		Mat out = Mat::zeros(im.size(), im.type());

		//singleScalePaint(im, out, Mat::ones(im.rows, im.cols, CV_32F), brush, 10, 5000);
		singleScalePaint(im, out, importance, brush, 10, 5000);
	
		//imshow("out", out);

		//waitKey();
	//}
	return 1;
}


// Brush and im must have 3 channels
bool applyStroke(Mat& im, int y, int x, Vec3f rgb, const Mat& brush) {
	if (im.channels() != 3 || brush.channels() != 3)
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
		for (int j = 0; j < 3*w; j+=3) {
			/*for (int c = 0; c < 3; c++) {
			stencil_ptr[j+c] = stencil_ptr[j+c]*(1-brush_ptr[j+c]) + brush_ptr[j+c]*rgb[c];
			}*/
			stencil_ptr[j+0] = stencil_ptr[j+0]*(1-brush_ptr[j+0]) + brush_ptr[j+0]*rgb[0];
			stencil_ptr[j+1] = stencil_ptr[j+1]*(1-brush_ptr[j+1]) + brush_ptr[j+1]*rgb[1];
			stencil_ptr[j+2] = stencil_ptr[j+2]*(1-brush_ptr[j+2]) + brush_ptr[j+2]*rgb[2];
		}
	}

	return true;
}

void singleScalePaint(const Mat& im, Mat& out, const Mat& importance, const Mat& brush, int size, int N, float noise) {
	int count = 0;

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

int sharpnessMap(const Mat& im, Mat& out, float sigma) {
	Mat im_grey;

	if (im.channels() == 1)
		im_grey = im.clone();
	else 
		cvtColor(im, im_grey, CV_BGR2GRAY);

	if (out.empty()) {
		out.create(im_grey.size(), im_grey.type());
	}

	GaussianBlur(im_grey, out, Size(9,9), sigma, sigma);

	out = im_grey - out;
	out = out.mul(out);

	GaussianBlur(out, out, Size(9,9), 4*sigma, 4*sigma);
	normalize(out);

	return 1;
}

void normalize(Mat& im) {
	double min, max;
	minMaxIdx(im, &min, &max, 0, 0);

	if ( (max - min) < 1e-4 )
		return; 

	im /= max;

	//im = (im - min) / (max - min);
}

void computerTensor(const cv::Mat& im, vector<float>& tensor, float sigma) {
	int h = im.rows;
	int w = im.cols;
	int N = h*w; 
	tensor.resize(N*3);

	Mat im_g;

	if (im.channels() == 1)
		im_g = im.clone();
	else 
		cvtColor(im, im_g, CV_BGR2GRAY);

	im_g.mul(im_g);
	GaussianBlur(im_g, im_g, Size(9,9), sigma, sigma);

	Mat dx, dy;

	Sobel(im_g, dx, -1, 1, 0);
	Sobel(im_g, dy, -1, 0, 1);

	Mat dxx, dyx, dyy;

	dxx = dx.mul(dx);
	dyx = dx.mul(dx);
	dyy = dy.mul(dy);

	int index = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tensor[index++] = dxx.at<float>(i,j);
			tensor[index++] = dyx.at<float>(i,j);
			tensor[index++] = dyy.at<float>(i,j);
		}
	}
}

void computeOrientation(const vector<float>& tensor, Mat& or) {
	vector<float> eig(2*tensor.size());

	const float* tensor_ptr = tensor.data();
	float* eig_ptr = eig.data();

	int N = tensor.size() / 3;

	eigen2x2(tensor_ptr, eig_ptr, N);

	int offset = 6*sizeof(float);
	float e1, e2, arg;
	for (int i = 0; i < or.rows; i++) {
		for (int j = 0; j < or.cols; j++) {
			arg = atan2f(eig_ptr[offset], eig_ptr[offset+1]);
			if (arg < 0)
				arg += 2*3.1415926535;
			arg /= 2*3.1415926535;
			or.at<float>(i,j) = arg;
		}
	}
}