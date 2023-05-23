#include "filters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;



Filters::Filters()
{
}




// Add salt and pepper noise to the image
Mat Filters::AddSaltPepperNoise(Mat &originalimage, float pa, float pb)
{
    RNG rng;    // random number generator
    vector<Mat>channels;
    split(originalimage, channels);
    int type= originalimage.type();
    int no_rows= channels[0].rows;
    int no_cols= channels[0].cols;
    int saltamount=no_rows*no_cols*pa; // the no. of pixels that contain salt noise
    int pepperamount=no_rows*no_cols*pb; // the no. of pixels that contain pepper noise


    if(type==0){

        for(int counter=0; counter<saltamount; ++counter)
         {
            originalimage.at<uchar>(rng.uniform( 0,originalimage.rows), rng.uniform(0, originalimage.cols)) =0;  //fill random pixels with 0 (salt noise)
         }

        for (int counter=0; counter<pepperamount; ++counter)
         {
            originalimage.at<uchar>(rng.uniform(0,originalimage.rows), rng.uniform(0,originalimage.cols)) =255; //fill random pixels with 255 (pepper noise)
         }
    }

    else{

        for (int i=0;i<=2;i++){
            for(int counter=0; counter<saltamount; ++counter)
             {
                channels[i].at<uchar>(rng.uniform( 0,originalimage.rows), rng.uniform(0, originalimage.cols)) =0;  //fill random pixels with 0 (salt noise)
             }

            for (int counter=0; counter<pepperamount; ++counter)
             {
                channels[i].at<uchar>(rng.uniform(0,originalimage.rows), rng.uniform(0,originalimage.cols)) =255; //fill random pixels with 255 (pepper noise)
             }
        }

        merge(channels,originalimage);
    }

    cout<< "S";

    return originalimage;
}





// Add gaussian noise to the image
Mat Filters::AddGaussianNoise(Mat &originalimage, double mean, double sd)   // mean , standard deviation(sigma)
{
    RNG rng;
    Mat NoiseArr= originalimage.clone();    // create copy of the original image        //0*
    rng.fill(NoiseArr, RNG::NORMAL, mean, sd);       // fill the created array with gaussian noise
    add(originalimage, NoiseArr, originalimage);   //originalimage+NoiseArr->originalimage

    cout<< "G";

    return originalimage;
}






// Add uniform noise to the image
Mat Filters::AddUniformNoise(Mat &originalimage, float a, float b)
{
    RNG rng;
    Mat NoiseArr= originalimage.clone();    // create copy of the original image
    rng.fill(NoiseArr, RNG::UNIFORM, a, b);       // fill the created array with gaussian noise
    add(originalimage, NoiseArr, originalimage);   //originalimage+NoiseArr->originalimage
    cout<< "U";

    return originalimage;
}





// member function to pad the image before convolution
Mat Filters::padding(Mat img, int kernalSize)
{
    Mat scr;
    img.convertTo(scr, CV_64FC1);
    int pad_rows, pad_cols;
    pad_rows = (kernalSize - 1) / 2;
    pad_cols = (kernalSize - 1) / 2;
    Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
    scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows)));
    return pad_image;

}





// member function to define kernels for convolution
Mat Filters::Filter (int kernalSize,int sigma, string type)
{
    // average kernel
    if (type == "average")
    {   double sum = 0;
        Mat kernel(kernalSize, kernalSize, CV_64FC1, Scalar(1.0 / (kernalSize * kernalSize)));
        return kernel;
    }

    // gaussian kernel
    else if (type == "gaussian")
    {   Mat kernel(kernalSize, kernalSize, CV_64FC1);
        double sum=0.0;
        int i,j;

        for (i=0 ; i<kernalSize ; i++) {
            for (j=0 ; j<kernalSize ; j++) {
                 kernel.at<double>(i, j)= exp(-(i*i+j*j)/(2*sigma*sigma))/(2*CV_PI*sigma*sigma);
                sum +=  kernel.at<double>(i, j);
            }
        }

        for (i=0 ; i<kernalSize ; i++) {
            for (j=0 ; j<kernalSize ; j++) {
                 kernel.at<double>(i, j) /= sum;
            }
        }

        return kernel;
    }
}






// member function to implement convolution
Mat Filters::convolve(Mat original,  int kernalSize,int sigma, string filterType)
{
    Mat pad_img, kernel;
    pad_img = padding(original,  kernalSize);
    kernel = Filter(kernalSize,sigma, filterType);

    Mat output = Mat::zeros(original.size(), CV_64FC1);

    for (int i = 0; i < original.rows; i++)
    {
        for (int j = 0; j < original.cols; j++)
        {
            output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, kernalSize,kernalSize)))).val[0];
        }
    }

    output.convertTo(output, CV_8UC1);
    return output;
}





Mat Filters::globalThresholding(Mat original,  int thresholdValue , int maxValue)
{
    Mat result = original.clone();
    for (int i = 0; i < original.rows; i++) {
        for (int j = 0; j < original.cols; j++) {
            if (original.at<uchar>(i, j) > thresholdValue) {
                result.at<uchar>(i, j) = maxValue;
            }
            else {
                result.at<uchar>(i, j) = 0;
            }
        }
    }

    return result;
}




// c is a variable of double type representing the constant subtracted from the mean or weighted mean.
Mat Filters::localThresholding(Mat original, int bloclSize, int c)
{
    Mat result = original.clone();
    int halfWindowSize = bloclSize / 2;
    for (int i = 0; i < original.rows ; i++) {
        for (int j = 0; j < original.cols ; j++) {
            int sum = 0;
            for (int n = -halfWindowSize; n <= halfWindowSize; n++) {
                for (int m = -halfWindowSize; m <= halfWindowSize; m++) {
                    sum += original.at<uchar>(i + n, j + m);
                }
            }
            int mean = sum / (bloclSize * bloclSize);
            if (original.at<uchar>(i, j) > mean - c) {
                result.at<uchar>(i, j) = 255;
            }
            else {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}





Mat Filters::medianKernal(Mat &image,int kernalSize)
{
    Mat new_image(image.rows, image.cols, CV_8U);

    for (int i = 0; i < image.rows;i++)
    {
        for (int j = 0; j < image.cols; j++)
        {

            int pad_rows = (kernalSize - 1) / 2;
            int pad_cols = (kernalSize - 1) / 2;
            int median = 0;
            int count = 0;
            int *arr = new int[kernalSize*kernalSize];
            for (int m = -pad_rows; m <= pad_rows; m++)
            {
                for (int n = -pad_cols; n <= pad_cols; n++)
                {
                    arr[count] = image.at<uchar>(i + m, j + n);
                    count++;
                }
            }
            sort(arr, arr + kernalSize*kernalSize);
            median = arr[(kernalSize*kernalSize) / 2];
            new_image.at<uchar>(i, j) = median;
        }
    }

    return new_image;
}



void Filters::generate_prewitt_kernel(int size, vector<vector<int>>& prewitt_x, vector<vector<int>>& prewitt_y)
{
    prewitt_x.resize(size, vector<int>(size));
    prewitt_y.resize(size, vector<int>(size));
    int middle = size / 2;
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (i == middle)
            {
                prewitt_y[i][j] = 0;
                prewitt_x[j][i] = 0;
            }
            else if (i < middle)
            {
                prewitt_y[i][j] = -1;
                prewitt_x[j][i] = -1;

            }
            else
            {
                prewitt_y[i][j] = 1;
                prewitt_x[j][i] = 1;

            }
        }
    }
}







void Filters::generateSobelKernels(int size, vector<vector<int>> &sobel_x, vector<vector<int>> &sobel_y)
{
    sobel_x.resize(size, vector<int>(size));
    sobel_y.resize(size, vector<int>(size));
    int center = size / 2;
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (i == center)
            {
                sobel_x[i][j] = 0;
                sobel_y[i][j] = 0;
            }
            else if (i < center)
            {
                sobel_x[i][j] = -1;
                sobel_y[i][j] = (j < center) ? -1 : 1;
            }
            else
            {
                sobel_x[i][j] = 1;
                sobel_y[i][j] = (j < center) ? 1 : -1;
            }
        }
    }
}






Mat Filters::applyGaussianFilter(const Mat &inputImage, int kernelSize, double sigma)
{
    Mat kernel = getGaussianKernel(kernelSize, sigma, CV_32F);
    Mat filteredImage;
    filter2D(inputImage, filteredImage, CV_32F, kernel);
    return filteredImage;
}





Mat Filters::applySobelFilter(const Mat &inputImage, int kernelSize, bool horizontal)
{
    Mat kernel;
    if (horizontal)
    {
        kernel = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    }
    else
    {
        kernel = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    }
    Mat filteredImage;
    filter2D(inputImage, filteredImage, CV_32F, kernel);
    return filteredImage;
}






Mat Filters::nonMaxSuppression(const Mat &magnitude, const Mat &direction)
{
    Mat suppressedImage(magnitude.rows, magnitude.cols, CV_32F, Scalar(0));
    float angle, a, b, c, d;

    for (int i = 1; i < magnitude.rows - 1; i++)
    {
        for (int j = 1; j < magnitude.cols - 1; j++)
        {
            angle = direction.at<float>(i, j);

            if ((angle < 22.5 && angle >= 0) || (angle >= 157.5 && angle < 202.5) || (angle >= 337.5 && angle <= 360))
            {
                a = magnitude.at<float>(i, j - 1);
                b = magnitude.at<float>(i, j + 1);
            }
            else if ((angle >= 22.5 && angle < 67.5) || (angle >= 202.5 && angle < 247.5))
            {
                a = magnitude.at<float>(i - 1, j + 1);
                b = magnitude.at<float>(i + 1, j - 1);
            }
            else if ((angle >= 67.5 && angle < 112.5) || (angle >= 247.5 && angle < 292.5))
            {
                a = magnitude.at<float>(i - 1, j);
                b = magnitude.at<float>(i + 1, j);
            }
            else
            {
                a = magnitude.at<float>(i - 1, j - 1);
                b = magnitude.at<float>(i + 1, j + 1);
            }

            c = magnitude.at<float>(i, j);
            if (c > a && c > b)
            {
                suppressedImage.at<float>(i, j) = c;
            }
        }
    }

    return suppressedImage;
}






// Function to apply double thresholding to the input image
void Filters::doubleThreshold(Mat &image, double lowThreshold, double highThreshold)
{
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            if (image.at<float>(i, j) < lowThreshold)
            {
                image.at<float>(i, j) = 0;
            }
            else if (image.at<float>(i, j) > highThreshold)
            {
                image.at<float>(i, j) = 255;
            }
            else
            {
                image.at<float>(i, j) = 128;
            }
        }
    }
}









Mat Filters::prewittEdgeDetector( Mat inputImage, int kernelSize)
{
    vector<vector<int>> kernelX, kernelY;
    generate_prewitt_kernel(kernelSize, kernelX, kernelY);

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Initialize output image with zeros
   Mat outputImage = Mat::zeros(height, width, CV_8UC1);

    // Loop over the image pixels
    for (int i = 0; i < height - kernelSize + 1; i++)
    {
        for (int j = 0; j < width - kernelSize + 1; j++)
        {
            // Compute the x and y gradients using the Prewitt kernels
            int gx = 0, gy = 0;
            for (int k = 0; k < kernelSize; k++)
            {
                for (int l = 0; l < kernelSize; l++)
                {

                    gx += inputImage.at<uchar>(i+k, j+l) * kernelX[k][l];
                    gy += inputImage.at<uchar>(i+k, j+l) * kernelY[k][l];
                }
            }

            // Compute the magnitude of the gradient
            int magnitude = std::sqrt(gx*gx + gy*gy);

            // Set the output pixel to the magnitude of the gradient
            outputImage.at<uchar>(i+kernelSize/2, j+kernelSize/2) = magnitude;
        }
    }
    return  outputImage;
}







Mat Filters::robert (Mat image)
{
    int kernal_x[2][2] = { { 1, 0 }, { 0, -1 } };
    int kernal_y[2][2] = { { 0, 1 }, { -1, 0 } };
    // Convert the input image to grayscale
    Mat gray_image;
    convertToGray(image, gray_image);
    // Apply Sobel filter
    int width = gray_image.cols;
    int height = gray_image.rows;
    int center = 2 / 2;
   Mat output_image = Mat::zeros(height, width, CV_8UC1);
    for (int i = center; i < height - center; ++i)
    {
        for (int j = center; j < width - center; ++j)
        {
            int grad_x = 0;
            int grad_y = 0;
            for (int k = 0; k < 2; ++k)
            {
                for (int l = 0; l < 2; ++l)
                {
                    int pixel_val = gray_image.at<uchar>(i + k - center, j + l - center);
                    grad_x += pixel_val * kernal_x[k][l];
                    grad_y += pixel_val * kernal_y[k][l];
                }
            }
            int grad = abs(grad_x) + abs(grad_y);
            output_image.at<uchar>(i, j) = grad;
        }
    }
    return output_image;
}








// edge
void Filters::convertToGray(const Mat &input_image, Mat &output_image)
{
    if (input_image.channels() == 3)
    {
        cvtColor(input_image, output_image, COLOR_BGR2GRAY);
    }
    else
    {
        output_image = input_image.clone();
    }
}








Mat Filters::cannyEdgeDetector( Mat input_image,  double lowThreshold, double highThreshold, int kernel_size, double sigma)
{
    // Convert the input image to grayscale
    Mat gray_image;
    convertToGray(input_image, gray_image);

    // Apply Gaussian filter
    Mat smoothed_image = applyGaussianFilter(gray_image, kernel_size, sigma);

    // // Apply Sobel filter to get gradients in x and y directions
    Mat grad_x = applySobelFilter(smoothed_image, 3, true);
    Mat grad_y = applySobelFilter(smoothed_image, 3, false);

    // Compute magnitude and direction of gradients
    Mat magnitude, direction;
    magnitude = Mat::zeros(grad_x.rows, grad_x.cols, CV_32F);
    direction = Mat::zeros(grad_x.rows, grad_x.cols, CV_32F);

    for (int i = 0; i < grad_x.rows; ++i)
    {
        for (int j = 0; j < grad_x.cols; ++j)
        {
            float gx = grad_x.at<float>(i, j);
            float gy = grad_y.at<float>(i, j);
            magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
            direction.at<float>(i, j) = atan2(gy, gx) * 180 / M_PI;
        }
    }

    // Apply non-maximum suppression
    Mat suppressed_image = nonMaxSuppression(magnitude, direction);

    // Apply double thresholding
    doubleThreshold(suppressed_image, lowThreshold, highThreshold);

    // Apply edge tracking by hysteresis
    Mat output_image = Mat::zeros(suppressed_image.rows, suppressed_image.cols, CV_8UC1);
    for (int i = 0; i < suppressed_image.rows; ++i)
    {
        for (int j = 0; j < suppressed_image.cols; ++j)
        {
            if (suppressed_image.at<float>(i, j) == 255)
            {
                // Start tracking the edge
                output_image.at<uchar>(i, j) = 255;

                // Check neighbors to see if they are also edges
                for (int ii = i - 1; ii <= i + 1; ++ii)
                {
                    for (int jj = j - 1; jj <= j + 1; ++jj)
                    {
                        if (ii >= 0 && ii < suppressed_image.rows && jj >= 0 && jj < suppressed_image.cols)
                        {
                            if (suppressed_image.at<float>(ii, jj) == 128)
                            {
                                // Mark this neighbor as an edge and continue tracking
                                suppressed_image.at<float>(ii, jj) = 255;
                                output_image.at<uchar>(ii, jj) = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    return output_image;
}






Mat Filters::sobelFilter( Mat input_image,  int kernel_size)
{
    // Convert the input image to grayscale
    Mat gray_image;
      convertToGray(input_image, gray_image);
    // Generate Sobel kernels
    vector<vector<int>> sobel_x, sobel_y;
    generateSobelKernels(kernel_size, sobel_x, sobel_y);

    // Apply Sobel filter
    int width = gray_image.cols;
    int height = gray_image.rows;
    int center = kernel_size / 2;
   Mat output_image = Mat::zeros(height, width, CV_8UC1);
    for (int i = center; i < height - center; ++i)
    {
        for (int j = center; j < width - center; ++j)
        {
            int grad_x = 0;
            int grad_y = 0;
            for (int k = 0; k < kernel_size; ++k)
            {
                for (int l = 0; l < kernel_size; ++l)
                {
                    int pixel_val = gray_image.at<uchar>(i + k - center, j + l - center);
                    grad_x += pixel_val * sobel_x[k][l];
                    grad_y += pixel_val * sobel_y[k][l];
                }
            }
            int grad = abs(grad_x) + abs(grad_y);
            output_image.at<uchar>(i, j) = grad;
        }
    }
    return output_image;
}








