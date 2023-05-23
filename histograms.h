#ifndef HISTOGRAMS_H
#define HISTOGRAMS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;



class Histograms
{
public:
    Histograms();
    void expand_img_to_optimal(Mat &padded, Mat &image);
    void crop_and_rearrange(Mat &magI);
    void DFT(Mat &image, Mat &output);
    void invers_DFT(Mat &DFT_image, Mat &imgOutput);
    void highpassFilter(Mat &dft_filter, int distance);
    void lowpassFilter(Mat &dft_filter, int distance);
    Mat Hyprid_images(Mat image1,Mat image2);
};

#endif // HISTOGRAMS_H
