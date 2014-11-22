#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "Painter.h"
#include "header.h"


using namespace cv;
using namespace std;

#define SHOW(a) std::cout << #a << "= " << endl << (a) << std::endl
#define PI 3.1415926535
#define TWOPI 2*PI

// Steps:
//  - seperate into high + low frequencies
//  - sample high, low
//  - 
int main(int argc, char* argv[]) {

	Mat brush;
	if (argc > 1)
		brush = imread(argv[1]);  
	else
		brush = imread("longBrush2.png");

	if (argc > 2 && string(argv[2]) == "--novid") {
		Paint(brush, 60, true);
	} else if (argc > 2 && string(argv[2]) == "--onestroke") {
		PaintStrokeByStroke(brush);
	} else {
		Paint(brush, 60);
	}

	return 0;
}

void Paint(Mat brush, int size, bool novid, int N, int nAngles, float noise) {
	brush.convertTo(brush, CV_32F, 1/255.);

	if (brush.channels() == 3)
		cvtColor(brush, brush, CV_BGR2GRAY);

	Painter p(brush, nAngles);

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

		p.Paint(frame, out);

		imshow("vid", out);
		if(waitKey(1) >= 0) break;
	}

	return;
}

void PaintStrokeByStroke(Mat brush) {
	brush.convertTo(brush, CV_32F, 1/255.);

	if (brush.channels() == 3)
		cvtColor(brush, brush, CV_BGR2GRAY);

	Painter p(brush, 36);

	namedWindow("vid", WINDOW_NORMAL);
	Mat frame = imread("chase.jpg");
	frame.convertTo(frame, CV_32F, 1/255.);

	Mat out = Mat::zeros(frame.size(), CV_32FC3);
	for (int i = 0; i < 5000; i++)
	{
		p.PaintStrokeByStroke(frame, out);
		imshow("vid", out);
		if(waitKey(1) >= 0) break;
	}
	for (int i = 0; i < 5000; i++)
	{
		p.PaintStrokeByStroke(frame, out, true);
		imshow("vid", out);
		if(waitKey(1) >= 0) break;
	}

	waitKey(0);
	return;
}

