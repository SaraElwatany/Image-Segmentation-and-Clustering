#ifndef FREQUENCY_H
#define FREQUENCY_H


#include <QFileDialog>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


class Frequency
{


public:
    Frequency();
    Mat GetHistogram(Mat originalimage);
    Mat CumulativeHistogram(Mat histogram);
    Mat DrawHistogram(Mat histogram);
    Mat Equalize(Mat originalimage);
    Mat ImageSplit(Mat originalimage,int type);
    Mat GreyScale(Mat originalimage);
    Mat Normalize(Mat originalimage);
    uchar getMinMax(Mat image);


private:


};

#endif // FREQUENCY_H
