#ifndef ACTIVECONTOUR_H
#define ACTIVECONTOUR_H


#include <opencv2/opencv.hpp>

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace cv;

class ActiveContour
{
public:
    ActiveContour();
    std::vector<Point> initial_contour(cv::Point center,int radius);
    double   calcInternalEnergy(cv::Point pt, cv::Point prevPt, cv::Point nextPt, double alpha);
    double   calcExternalEnergy(Mat img, cv::Point pt, double beta);
    double calcBalloonEnergy(cv::Point pt, cv::Point prevPt, double gamma) ;
    std::vector<Point>  contourUpdating(Mat inputImg, Mat &outputImg,
                        Point center, int radius,
                        int numOfIterations,
                        double alpha, double beta, double gamma);
    std::vector<int>  calculateChainCode(const std::vector<Point>& contour);


};

#endif // ACTIVECONTOUR_H
