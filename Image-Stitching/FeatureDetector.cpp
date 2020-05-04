#include "FeatureDetector.h"

void buildPyramid(std::vector<cv::Mat>& pyramid, int _scale, int _level, float _sigma)
{
    for (int lvl = _level - 2; lvl >= 0; lvl--) {
        // apply gaussian blur
        cv::GaussianBlur(pyramid[lvl + 1], pyramid[lvl], cv::Size(3, 3), _sigma);
        // downsample
        cv::Size dsize = cv::Size(pyramid[lvl + 1].cols / _scale, pyramid[lvl + 1].rows / _scale);
        cv::resize(pyramid[lvl + 1], pyramid[lvl], dsize);
        /*cv::imshow("pyramid" + std::to_string(lvl + 1) + ".jpg", pyramid[lvl]);*/
    }
}

void computeGradient(cv::Mat src, cv::Mat& dst, int xOrder, int yOrder)
{
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            float tmp1 = 0.0, tmp2 = 0.0;
            if ((xOrder == 0) && (yOrder == 1)) {
                if (y - 1 >= 0)
                    tmp1 = src.at<uchar>(y - 1, x);
                if (y + 1 < dst.rows)
                    tmp2 = src.at<uchar>(y + 1, x);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 2;
            }
            else if ((xOrder == 1) && (yOrder == 0)) {
                if (x - 1 >= 0)
                    tmp1 = src.at<uchar>(y, x - 1);
                if (x + 1 < dst.cols)
                    tmp2 = src.at<uchar>(y, x + 1);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 2;
                /*std::cout << (int)dst.at<uchar>(y, x) << std::endl;*/
            }
            else if ((xOrder == 1) && (yOrder == 1)) {
                if ((x - 1 >= 0) && (y - 1 >= 0))
                    tmp2 = src.at<uchar>(y - 1, x - 1);
                if ((x + 1 < dst.cols) && (y + 1 < dst.rows))
                    tmp2 += src.at<uchar>(y + 1, x + 1);
                if ((x - 1 >= 0) && (y + 1 < dst.rows))
                    tmp1 = src.at<uchar>(y + 1, x - 1);
                if ((x + 1 < dst.cols) && (y - 1 >= 0))
                    tmp1 += src.at<uchar>(y - 1, x + 1);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 4;
            }
            else if ((xOrder == 0) && (yOrder == 2)) {
                if (y - 1 >= 0)
                    tmp1 = src.at<uchar>(y - 1, x);
                if (y + 1 < dst.rows)
                    tmp2 = src.at<uchar>(y + 1, x);
                dst.at<uchar>(y, x) = tmp2 - 2 * src.at<uchar>(y, x) + tmp1;
            }
            else if ((xOrder == 2) && (yOrder == 0)) {
                if (x - 1 >= 0)
                    tmp1 = src.at<uchar>(y, x - 1);
                if (x + 1 < dst.cols)
                    tmp2 = src.at<uchar>(y, x + 1);
                dst.at<uchar>(y, x) = tmp2 - 2 * src.at<uchar>(y, x) + tmp1;
            }
        }
    }
}

void computeCornerResponse(std::vector<cv::Mat> pyramid, std::vector<cv::Mat>& response, int _level, float _sigma_d, float _sigma_i)
{
    cv::Mat Ix, Iy, Ix2, Iy2, IxIy;
    for (int lvl = 0; lvl < _level; lvl++) {
        cv::Mat img;
        pyramid[lvl].copyTo(img);
        // compute Ix, Iy of the image at each level of the pyramid
        pyramid[lvl].copyTo(Ix);
        pyramid[lvl].copyTo(Iy);
        // blurred x-direction/y-direction gradient -> Ix, Iy
        computeGradient(pyramid[lvl], img, 1, 0);
        cv::GaussianBlur(img, Ix, cv::Size(3, 3), _sigma_d);
        /*cv::imshow("gx" + std::to_string(lvl + 1) + ".jpg", Ix);
        cv::waitKey(1);*/
        computeGradient(pyramid[lvl], img, 0, 1);
        cv::GaussianBlur(img, Iy, cv::Size(3, 3), _sigma_d);
        /*cv::imshow("gy" + std::to_string(lvl + 1) + ".jpg", Iy);
        cv::waitKey(1);*/
        // compute Ix2, Iy2, and Ixy (product of derivatives) at each level of the pyramid
        pyramid[lvl].copyTo(Ix2);
        pyramid[lvl].copyTo(Iy2);
        pyramid[lvl].copyTo(IxIy);
        // squared & blurred gradients -> Ix2, Iy2, Ixy
        for (int i = 0; i < pyramid[lvl].rows; i++)
            for (int j = 0; j < pyramid[lvl].cols; j++)
                img.at<uchar>(i, j) = Ix.at<uchar>(i, j) * Ix.at<uchar>(i, j);
        cv::GaussianBlur(img, Ix2, cv::Size(3, 3), _sigma_i);
        for (int i = 0; i < pyramid[lvl].rows; i++)
            for (int j = 0; j < pyramid[lvl].cols; j++)
                img.at<uchar>(i, j) = Iy.at<uchar>(i, j) * Iy.at<uchar>(i, j);
        cv::GaussianBlur(img, Iy2, cv::Size(3, 3), _sigma_i);
        for (int i = 0; i < pyramid[lvl].rows; i++)
            for (int j = 0; j < pyramid[lvl].cols; j++)
                img.at<uchar>(i, j) = Ix.at<uchar>(i, j) * Iy.at<uchar>(i, j);
        cv::GaussianBlur(img, IxIy, cv::Size(3, 3), _sigma_i);
        // compute Harris corner response
        // M = [ Ix2   IxIy ]
        //     [ IxIy  Iy2  ]
        // det(M) = |M| = Ix2 * Iy2 - IxIy * IxIy
        // tr(M) = sum of diagonal = Ix2 + Iy2
        for (int i = 0; i < response[lvl].rows; i++) {
            for (int j = 0; j < response[lvl].cols; j++) {
                response[lvl].at<uchar>(i, j) = 255 * 255; // cv::Mat stores data between 0 and 1
                
                if (Ix2.at<uchar>(i, j) + Iy2.at<uchar>(i, j) == 0.0)
                    response[lvl].at<uchar>(i, j) = 0.0;
                else
                    response[lvl].at<uchar>(i, j) *= ((Ix2.at<uchar>(i, j) * Iy2.at<uchar>(i, j)) - (IxIy.at<uchar>(i, j) * IxIy.at<uchar>(i, j))) / (Ix2.at<uchar>(i, j) + Iy2.at<uchar>(i, j));
            }
        }
        /*cv::imshow("resp" + std::to_string(lvl + 1) + ".jpg", response[lvl]);
        cv::waitKey(1);*/
    }
}

void computeOrientation(const std::vector<cv::Mat>& pyramid, std::vector<cv::Mat>& orientation, int _level, float _sigma_o)
{
    cv::Mat Ix, Iy, img;
    for (int lvl = 0; lvl < _level; lvl++) {
        pyramid[lvl].copyTo(img);
        pyramid[lvl].copyTo(Ix);
        pyramid[lvl].copyTo(Iy);
        // blurred x-direction gradient -> Ix
        computeGradient(pyramid[lvl], img, 1, 0);
        cv::GaussianBlur(img, Ix, cv::Size(3, 3), _sigma_o);
        // blurred y-direction gradient  -> Iy
        computeGradient(pyramid[lvl], img, 0, 1);
        cv::GaussianBlur(img, Iy, cv::Size(3, 3), _sigma_o);
        // [cos(theta), sin(theta)] = [Ix, Iy] 
        // => theta = atan(Iy / Ix)
        for (int i = 0; i < orientation[lvl].rows; i++)
            for (int j = 0; j < orientation[lvl].cols; j++)
                orientation[lvl].at<char>(i, j) = atan2(Iy.at<char>(i, j), Ix.at<char>(i, j));
    }
}

void findFeaturesOnPyramid(const std::vector<cv::Mat>& response, bool** isFeature, int level, float threshold)
{
    for (int lvl = 0; lvl < level; lvl++) {
        int w = response[lvl].cols;
        int h = response[lvl].rows;

        int nFeatureCandidates = 0;  // # of points whose response > 10
        int nRejected = 0;  // # of jejected points whose response > 10  

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                isFeature[lvl][i * w + j] = false;
                // if responce greater than the threshold then it may be a feature
                if (response[lvl].at<char>(i, j) > threshold) {
                    isFeature[lvl][i * w + j] = true;
                    nFeatureCandidates++;
                    // check if (i, j) is the maxima point in the 3 x 3 region
                    for (int u = -1; u <= 1 && isFeature[lvl][i * w + j]; u++) {
                        if ((i + u < 0) || (i + u >= h))
                            continue;
                        for (int v = -1; v <= 1; v++) {
                            if ((j + v < 0) || (j + v >= w))
                                continue;
                            if (response[lvl].at<char>(i + u, j + v) > response[lvl].at<char>(i, j)) {
                                isFeature[lvl][i * w + j] = false;
                                nRejected++;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void subPixelAccuracy(const std::vector<cv::Mat>& res, bool** isFeature, int level)
{
    cv::Mat dfdx1, dfdy1, dfdx2, dfdy2, dfdxy;
    // work, work
    for (int lvl = 0; lvl < level; lvl++) {
        int w = res[lvl].cols;
        int h = res[lvl].rows;

        res[lvl].copyTo(dfdx1);
        res[lvl].copyTo(dfdy1);
        res[lvl].copyTo(dfdx2);
        res[lvl].copyTo(dfdy2);
        res[lvl].copyTo(dfdxy);

        computeGradient(res[lvl], dfdx1, 1, 0);
        computeGradient(res[lvl], dfdy1, 0, 1);
        computeGradient(res[lvl], dfdx2, 2, 0);
        computeGradient(res[lvl], dfdy2, 0, 2);
        computeGradient(res[lvl], dfdxy, 1, 1);

        /* parse isFeature */
        std::vector<FeaturePoint> Pts;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (isFeature[lvl][i * w + j]) {
                    FeaturePoint Pt;
                    Pt.x = j;
                    Pt.y = i;
                    Pts.push_back(Pt);
                }
            }
        }
        /* parse done */
        // Xm = -[A]inverse * [B]  , A=d2f/d<x>2, B=df/d<x>
        int needShift = 0;   // # of features that need to be shifted
        for (int i = 0; i < Pts.size(); i++) {
            float A[2][2] = { { dfdx2.at<uchar>(Pts[i].y, Pts[i].x), dfdxy.at<uchar>(Pts[i].y, Pts[i].x) },
                               { dfdxy.at<uchar>(Pts[i].y, Pts[i].x), dfdy2.at<uchar>(Pts[i].y, Pts[i].x) } }; // A
            float detA = A[0][0] * A[1][1] - A[0][1] * A[1][0]; // det(A)
            float Ai[2][2] = { {  A[1][1] / detA, -A[0][1] / detA },
                               { -A[1][0] / detA,  A[0][0] / detA } }; // A inverse
            float B[2] = { dfdx1.at<uchar>(Pts[i].y, Pts[i].x), dfdy1.at<uchar>(Pts[i].y, Pts[i].x) }; // B
            float offset[2] = { -Ai[0][0] * B[0] - Ai[0][1] * B[1], -Ai[1][0] * B[0] - Ai[1][1] * B[1] }; // ans
            // if the offset if larger than 0.5, shift the sample point of the feature once
            if (offset[0] > 0.5 || offset[0] < -0.5 || offset[1]>0.5 || offset[1] < -0.5) {
                needShift++;
                /*make shift to isFeature map*/
                isFeature[lvl][Pts[i].x + Pts[i].y * w] = false;
                if (offset[0] > 0.5 && Pts[i].x + 1 < w)
                    Pts[i].x++;
                else if (offset[0] < -0.5 && Pts[i].x - 1 >= 0)
                    Pts[i].x--;
                if (offset[1] > 0.5 && Pts[i].y + 1 < h)
                    Pts[i].y++;
                else if (offset[1] < -0.5 && Pts[i].y - 1 >= 0)
                    Pts[i].y--;
                isFeature[lvl][Pts[i].x + Pts[i].y * w] = true;
            }
        }
    }
}

void getAllFeatures(std::vector<FeaturePoint>& features, FeaturePoint** featureMap, const std::vector<cv::Mat>& response, const std::vector<cv::Mat>& orientation, bool** isFeature, int _level, int _scale)
{
    // init
    for (int i = 0; i < response[_level - 1].rows; i++) {
        for (int j = 0; j < response[_level - 1].cols; j++) {
            //printf("%d %d\n", i, j);
            featureMap[i][j].level = -1;
        }
    }
    // project features on all levels 
    for (int lvl = _level - 1, s = 1; lvl >= 0; lvl--, s *= _scale) {
        int w = response[lvl].cols, h = response[lvl].rows;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                // (i, j) on lvl is a feature
                if (isFeature[lvl][i * w + j]) {
                    // (i*s, j*s) on level has no features projected to
                    if (featureMap[i * s][j * s].level == -1) {
                        featureMap[i * s][j * s].x = j * s;
                        featureMap[i * s][j * s].y = i * s;
                        featureMap[i * s][j * s].level = lvl;
                        featureMap[i * s][j * s].orientation = orientation[lvl].at<uchar>(i, j);
                        featureMap[i * s][j * s].response = response[lvl].at<uchar>(i, j);
                    }
                    // (i*s, j*s) on level already has a features projected to
                    else {
                        // if features of different scales project to the same pixel, preserve the one with largest response
                        if (response[lvl].at<uchar>(i, j) > featureMap[i * s][j * s].response) {
                            featureMap[i * s][j * s].level = lvl;
                            featureMap[i * s][j * s].orientation = orientation[lvl].at<uchar>(i, j);
                            featureMap[i * s][j * s].response = response[lvl].at<uchar>(i, j);
                        }
                    }
                }
            }
        }
    }
    // collect all projected features
    for (int i = 0; i < response[_level - 1].rows; i++)
        for (int j = 0; j < response[_level - 1].cols; j++)
            if (featureMap[i][j].level != -1)
                features.push_back(featureMap[i][j]);
}

void nonMaximalSuppression(std::vector<FeaturePoint>& features, int desiredNum, int initRadius, int step)
{
    int desiredNumFixed = (int)(desiredNum + NMS_TOLLERATE_RATIO * desiredNum);
    int currentNum = desiredNumFixed + 1;
    std::vector<bool> valid;
    valid.assign(features.size(), true);
    // work, work 
    for (int radius = initRadius; currentNum > desiredNumFixed; radius += step) {
        int radiusSquared = radius * radius;
        valid.assign(features.size(), true);
        for (int i = 0; i < features.size(); i++) {
            if (!valid[i])	continue;
            for (int j = 0; j < features.size(); j++) {
                if ((i == j)
                    || ((features[i].x - features[j].x) * (features[i].x - features[j].x) + (features[i].y - features[j].y) * (features[i].y - features[j].y)) >= radiusSquared)	//是自己或在圓外就不算 
                    continue;
                if (features[j].response < features[i].response)
                    valid[j] = false;
                else {
                    valid[i] = false;
                    break;
                }
            }
        }
        currentNum = (int)count(valid.begin(), valid.end(), true);
#ifdef DEBUG
        printf("\tradius = %d, current # of features = %d\n", radius, currentNum);
#endif
    }
    // delete features
    for (int i = 0; i < features.size(); i++) {
        if (!valid[i]) {
            features.erase(features.begin() + i);
            valid.erase(valid.begin() + i);
            i--;
        }
    }
}

void deleteFpTooCloseToBounds(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale)
{
    // init
    int* s = new int[_level];
    s[_level - 1] = 1;
    for (int lvl = _level - 2; lvl >= 0; lvl--)
        s[lvl] = s[lvl + 1] * _scale;
    // work, work
    for (int k = 0; k < features.size(); k++) {
        FeaturePoint fp = features[k];
        // rotate a 40 x 40 descriptor sampling window
        float rotation[2][2] = { {cos(fp.orientation), -sin(fp.orientation)},
                                {sin(fp.orientation),  cos(fp.orientation)} };
        float corner[4][2] = { { rotation[0][0] * (-20) + rotation[0][1] * (-20),
                                rotation[1][0] * (-20) + rotation[1][1] * (-20) }, // top left
                              { rotation[0][0] * (19) + rotation[0][1] * (-20),
                                rotation[1][0] * (19) + rotation[1][1] * (-20) }, // top right
                              { rotation[0][0] * (-20) + rotation[0][1] * (19),
                                rotation[1][0] * (-20) + rotation[1][1] * (19) }, // bottom left
                              { rotation[0][0] * (19) + rotation[0][1] * (19),
                                rotation[1][0] * (19) + rotation[1][1] * (19) } }; // bottom right
        // if part of the a feature's window falls out of image, delete it
        int x = (int)((float)fp.x / s[fp.level]);
        int y = (int)((float)fp.y / s[fp.level]);
        for (int i = 0; i < 4; i++) {
            if ((x + corner[i][0] < 0) || (y + corner[i][1] < 0)
                || (x + corner[i][0] >= pyramid[fp.level].cols)
                || (y + corner[i][1] >= pyramid[fp.level].rows)) {
                features.erase(features.begin() + k);
                k--;
                break;
            }
        }
    }
    delete[] s;
}

void computeFeatureDescriptor(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale)
{
    // init
    int* s = new int[_level];
    s[_level - 1] = 1;
    for (int lvl = _level - 2; lvl >= 0; lvl--)
        s[lvl] = s[lvl + 1] * _scale;
    // work, work
    for (int k = 0; k < features.size(); k++) {
        FeaturePoint fp = features[k];
        // rotate a 40 x 40 descriptor sample window
        float rotation[2][2] = { {cos(fp.orientation), -sin(fp.orientation)},
                                {sin(fp.orientation),  cos(fp.orientation)} };
        float window[40][40];
        for (int u = -20; u < 20; u++) {  // row
            for (int v = -20; v < 20; v++) {  // col
                int v_r = (int)(rotation[0][0] * u + rotation[0][1] * v);
                int u_r = (int)(rotation[1][0] * u + rotation[1][1] * v);
                window[u + 20][v + 20] =
                    pyramid[fp.level].at<uchar>((int)(((float)fp.y / s[fp.level]) + u_r), (int)(((float)fp.x / s[fp.level]) + v_r));
            }
        }
        // each element of 64D desrciptor is the mean of every 5 x 5 samples in the 40 x 40 window
        for (int i = 0; i < 40; i += 5) {  // row
            for (int j = 0; j < 40; j += 5) {  // col
                features[k].descriptor[(i / 5) * 8 + (j / 5)] = 0.0;
                for (int u = 0; u < 5; u++)  // row
                    for (int v = 0; v < 5; v++)  // col
                        features[k].descriptor[(i / 5) * 8 + (j / 5)] += window[i + u][j + v];
                features[k].descriptor[(i / 5) * 8 + (j / 5)] /= 25;
            }
        }
        // normalize the desrciptor to N(0, 1)
        float mean = 0.0;
        for (int i = 0; i < 64; i++)
            mean += features[k].descriptor[i];
        mean /= 64;
        float stddev = 0.0;
        for (int i = 0; i < 64; i++)
            stddev += ((features[k].descriptor[i] - mean) * (features[k].descriptor[i] - mean));
        stddev = sqrt(stddev / 64);
        for (int i = 0; i < 64; i++)
            features[k].descriptor[i] = (features[k].descriptor[i] - mean) / stddev;
    }
}

void drawFeatures(const std::vector<FeaturePoint>& features, cv::Mat& src) {
    cv::Mat copy;
    src.copyTo(copy);
    for (auto feature : features) {
        cv::circle(copy, cv::Point(feature.x, feature.y), 3, cv::Scalar(0, 0, 255));
    }
    //cv::imwrite("features.jpg", copy);
    cv::imshow("features.jpg", copy);
    cv::waitKey(1);
}

cv::Mat DetectFeature(cv::Mat src, std::vector<FeaturePoint> features)
{
    int level = 5;
    int scale = 2;
    float sigma = 1.0;
    float sigma_d = 1.0;
    float sigma_i = 1.5;
    float sigma_o = 4.5;
    float local_thresh = 10.0;

	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	std::vector<cv::Mat> pyramid(level);
    for (int i = 0; i < level; i++) {
        gray.copyTo(pyramid[i]);
    }
	buildPyramid(pyramid, scale, level, sigma);

    // 2-1: compute Harris corner response
    std::vector<cv::Mat> response(level);
    for (int i = 0; i < level; i++) {
        pyramid[i].copyTo(response[i]);
    }
    computeCornerResponse(pyramid, response, level, sigma_d, sigma_i);

    // 2-2: compute feature orientation
    std::vector<cv::Mat> orientation(level);
    for (int i = 0; i < level; i++) {
        pyramid[i].copyTo(orientation[i]);
    }
    computeOrientation(pyramid, orientation, level, sigma_o);

    // 2-3: find local maxima and thresholding -> interest points(features)
    bool** isFeature = new bool* [level];
    for (int lvl = 0; lvl < level; lvl++)
        isFeature[lvl] = new bool[response[lvl].cols * response[lvl].rows];
    findFeaturesOnPyramid(response, isFeature, level, local_thresh);

    // 2-4: sub-pixel accuracy
    subPixelAccuracy(response, isFeature, level);

    // Step 3: feature filter 
    // 3-1: project features back to original resolution image
    /*std::vector<FeaturePoint> features;*/
    FeaturePoint** featureMap = new FeaturePoint * [src.rows];
    for (int i = 0; i < src.rows; i++)
        featureMap[i] = new FeaturePoint[src.cols];
    getAllFeatures(features, featureMap, response, orientation, isFeature, level, scale);
    response.resize(0);
    orientation.resize(0);
    drawFeatures(features, src);
    /*drawFeatures(features, 0, false, level, n, inputName);*/
    //printf("\tTotal of %d features\n", features.size());
    int* counter = new int[level];
    for (int lvl = 0; lvl < level; lvl++)
        counter[lvl] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    for (int lvl = 0; lvl < level; lvl++)
        printf("\tlvl%2d: %d\n", lvl, counter[lvl]);

    // 3-2: throw away the features that are too close the boundaries on the corresponding scale
    printf("\nDeleting features too close to the image boundaries ...\n");
    deleteFpTooCloseToBounds(features, pyramid, level, scale);
    /*drawFeatures(features, 1, false, level, n, inputName);*/
    printf("\tTotal of %d features\n", features.size());
    for (int lvl = 0; lvl < level; lvl++)
        counter[lvl] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    for (int lvl = 0; lvl < level; lvl++)
        printf("\tlvl%2d: %d\n", lvl, counter[lvl]);

    // 3-3: non-maximal suppression
    printf("\nApply non-maximal suppression ...\n");
    nonMaximalSuppression(features, NMS_TOTAL_FEATURE_NUM, NMS_INIT_RADIUS, NMS_RADIUS_INCREASE_RATE);

    // (including oriented sample bwindow)
    for (int lvl = 0; lvl < level; lvl++)
        counter[lvl] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    printf("\tTotal of %d features\n", features.size());
    for (int lvl = 0; lvl < level; lvl++)
        printf("\tlvl%2d: %d\n", lvl, counter[lvl]);

    // Step 4: Feature descriptor
    printf("\nCompute feature descriptors ...\n");
    computeFeatureDescriptor(features, pyramid, level, scale);

    for (int i = 0; i < src.rows; i++)
        delete[] featureMap[i];
    delete[] featureMap;
    delete[] counter;
	return src;
}
