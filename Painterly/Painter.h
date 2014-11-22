#pragma once

#include <opencv2/core/core.hpp>
#include <random>

class Painter
{
public:
	Painter(void);
	Painter(const cv::Mat& brush, int numAngles = 36, int brushSize = -1);

	void Paint(cv::Mat frame, cv::Mat& out);
	void PaintLargeScale(cv::Mat frame, cv::Mat& out, bool oriented=false);
	void PaintSmallScale(cv::Mat frame, cv::Mat& out, bool oriented=false);

	int getNumAngles() { return m_numAngles; }
	int getStrokeSize() { return m_strokeSize; }
	float getTensorSigma1() { return m_tensorSigma1; }
	float getTensorSigma2() { return m_tensorSigma2; }
	float getSharpnessSigma() { return m_shaprnessSigma; }

	void setNumAngles(int numAngles) { m_numAngles = numAngles; updateBrush(); }
	void setStrokeSize(int strokeSize) { m_strokeSize = strokeSize; updateBrush(); }
	void setTensorSigma1(float tensorSigma1) { m_tensorSigma1 = tensorSigma1; }
	void setTensorSigma2(float tensorSigma2) { m_tensorSigma2 = tensorSigma2; }
	void setSharpnessSigma(float shaprnessSigma) { m_shaprnessSigma = shaprnessSigma; }

	// HACKS
	void PaintStrokeByStroke(const cv::Mat& frame, cv::Mat& out, bool small=false);
private:
	std::mt19937 m_mt;

	std::vector<cv::Mat> m_brushes;
	std::vector<cv::Mat> m_brushesSmall;

	int m_numAngles;
	int m_strokeSize;
	float m_tensorSigma1;
	float m_tensorSigma2;
	float m_shaprnessSigma;

	int m_numLargeStrokes;
	int m_numSmallStrokes;

	cv::Mat m_originalBrush;
	cv::Mat m_orientation;
	cv::Mat m_details;
	
	
	// Apply a single stroke at given location if possible
	bool applyStroke(cv::Mat& im, int y, int x, cv::Vec3f rgb, const cv::Mat& brush);

	void singleScalePaint(const cv::Mat& im, cv::Mat& out, const cv::Mat& importance, 
					  const cv::Mat& brush, int N=1000, float noise=0.3);

	void singleScaleOrientedPaint(const cv::Mat& im, cv::Mat& out, const cv::Mat& orientation, const cv::Mat& importance, 
					  const std::vector<cv::Mat>& brushes, int N=1000, float noise=0.3);

	// If we ever modify the brush, call this to update the rotated and resized brushes
	void updateBrush();

	// Given a brush stroke, transform it into numAngles images, each rotates 180/numAngles degrees.
	void computeRotations(const cv::Mat& brush, std::vector<cv::Mat>& rotations); 

	// Computes the orientation of the image at each pixel using the structure tensor
	void computeOrientation(const cv::Mat& frame, cv::Mat& orientation);
	
	// Computes the structure tensor
	void computerTensor(const cv::Mat& im, std::vector<float>& tensor, float sigma=3.0, float factor=5.0);
	
	// Computes the sharpness map of the image.
	int sharpnessMap(const cv::Mat& im, cv::Mat& out, float sigma=1.0);

	// Normalize an image to the range [0,1]
	void normalize(cv::Mat& im);

	// Given a sigma, calculate the necessary kernel size
	cv::Size computeKernelSize(float sigma);

};

