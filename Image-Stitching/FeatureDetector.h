#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Features.h"

#define NMS_TOTAL_FEATURE_NUM 500
#define NMS_RADIUS_INCREASE_RATE 1
#define NMS_INIT_RADIUS 8
#define NMS_TOLLERATE_RATIO 0.1

void buildPyramid(std::vector<cv::Mat>& pyramid, int _scale, int _level, float _sigma);
void computeGradient(cv::Mat src, cv::Mat& dst, int xOrder, int yOrder);
void computeCornerResponse(std::vector<cv::Mat> pyramid, std::vector<cv::Mat>& response, int _level, float _sigma_d, float _sigma_i);
void computeOrientation(const std::vector<cv::Mat>& pyramid, std::vector<cv::Mat>& orientation, int _level, float _sigma_o);
void findFeaturesOnPyramid(const std::vector<cv::Mat>& response, bool** isFeature, int level, float threshold);
void subPixelAccuracy(const std::vector<cv::Mat>& res, bool** isFeature, int level);
void getAllFeatures(std::vector<FeaturePoint>& features, FeaturePoint** featureMap, const std::vector<cv::Mat>& response, const std::vector<cv::Mat>& orientation, bool** isFeature, int _level, int _scale);
void nonMaximalSuppression(std::vector<FeaturePoint>& features, int desiredNum, int initRadius, int step);
void deleteFpTooCloseToBounds(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale);
void computeFeatureDescriptor(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale);
cv::Mat DetectFeature(cv::Mat src, std::vector<FeaturePoint>);
