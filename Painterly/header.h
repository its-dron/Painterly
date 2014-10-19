#include <opencv2\core\core.hpp>

bool applyStroke(cv::Mat& im, int y, int x, cv::Vec3f rgb, const cv::Mat& brush);

void singleScalePaint(const cv::Mat& im, cv::Mat& out, const cv::Mat& importance, 
					  const cv::Mat& brush, int size=10, int N=1000, float noise=0.3);

int sharpnessMap(const cv::Mat& im, cv::Mat& out, float sigma=1.0);

void normalize(cv::Mat& im);

void computerTensor(const cv::Mat& im, std::vector<float>& tensor, float sigma=3.0, float factor=5.0);

void computeOrientation(const std::vector<float>& tensor, cv::Mat& or, cv::Mat& mag);
