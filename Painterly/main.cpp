#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "Painter.h"
#include "header.h"

using namespace cv;
using namespace std;

Painter p;
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

	// TODO: add arg parsing
	Setup(brush, 60);

	if (argc > 2 && string(argv[2]) == "--novid") {
		Paint(true);
	} else {
		Paint(false);
	}

	return 0;
}

// Set up painter
void Setup(Mat brush, int size, int N, int nAngles, float noise) {
	brush.convertTo(brush, CV_32F, 1/255.);

	if (brush.channels() == 3)
		cvtColor(brush, brush, CV_BGR2GRAY);

	p = Painter(brush, nAngles, size);

	// TODO: add noise
}

// The main paint loop!
void Paint(bool novid) {

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


