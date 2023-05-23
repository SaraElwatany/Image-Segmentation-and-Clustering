#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


class Filters
{

public:
    Filters();
    Mat AddUniformNoise(Mat &originalimage, float a, float b);
    Mat AddGaussianNoise(Mat &originalimage, double mean, double sd);   // mean , standard deviation(sigma)
    Mat AddSaltPepperNoise(Mat &originalimage, float pa, float pb);
    Mat padding(Mat img, int kernalSize);
    Mat Filter (int kernalSize,int sigma, string type);
    Mat convolve(Mat original,  int kernalSize,int sigma, string filterType);
    Mat globalThresholding(Mat original,  int thresholdValue , int maxValue);
    Mat localThresholding(Mat original, int bloclSize, int c);
    Mat medianKernal(Mat &image,int kernalSize);
    void generate_prewitt_kernel(int size, vector<vector<int>>& prewitt_x, vector<vector<int>>& prewitt_y);
    Mat applyGaussianFilter(const Mat &inputImage, int kernelSize, double sigma);
    Mat applySobelFilter(const Mat &inputImage, int kernelSize, bool horizontal);
    Mat nonMaxSuppression(const Mat &magnitude, const Mat &direction);
    void doubleThreshold(Mat &image, double lowThreshold, double highThreshold);
    void generateSobelKernels(int size, vector<vector<int>> &sobel_x, vector<vector<int>> &sobel_y);
    Mat prewittEdgeDetector( Mat inputImage, int kernelSize);
    Mat robert (Mat image);
    Mat cannyEdgeDetector( Mat input_image,  double lowThreshold, double highThreshold, int kernel_size, double sigma);
    Mat sobelFilter( Mat input_image,  int kernel_size);
    void convertToGray(const Mat &input_image, Mat &output_image);


};

#endif // FILTERS_H
