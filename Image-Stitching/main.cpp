#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "Features.h"
#include "FeatureDetector.h"

int main() {
	cv::Mat image;
	image = cv::imread("E:\\assignment\\VFX\\Image-Stitching\\Image-Stitching\\1.JPG", CV_LOAD_IMAGE_COLOR);
	
	std::vector<FeaturePoint> features;
	auto feature_image = DetectFeature(image, features);
	//cv::namedWindow("Feature Image", 1);
	cv::imshow("Feature Image", image);
	cv::waitKey(0);

	return 0;
}
