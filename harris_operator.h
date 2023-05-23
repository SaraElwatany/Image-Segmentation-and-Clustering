#ifndef HARRIS_OPERATOR_H
#define HARRIS_OPERATOR_H

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
using namespace std;
using namespace cv;

class harris_operator
{
public:
    harris_operator();
    void nonMaxSuppression(Mat& response, int window_size);
    Mat convertToGray(Mat img) ;
    Mat harrisCorner(Mat gray,int window_size,double alpha);
    void setThreshold(Mat& R,double threshold_level) ;
    void getLocalMax(Mat R, Mat& localMax);
    Mat  drawPointsToImg(Mat img, Mat localMax);

};

#endif // HARRIS_OPERATOR_H

