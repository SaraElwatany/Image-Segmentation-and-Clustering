#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;


class Functions
{

public:
    Functions();

private:

    Mat GetHistogram(Mat originalimage);
    Mat CumulativeHistogram(Mat histogram);
    void DrawHistogram(Mat histogram, string window_name);
    void Equalize(Mat originalimage);
    void Normalize(Mat originalimage, float value);


};

#endif // FUNCTIONS_H
