#include "functions.h"


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;


Functions::Functions()
{
 }



// Calculate the values at histogram bins
Mat Functions::GetHistogram(Mat originalimage)
{
    int bin= 0;
    Mat histogram = Mat::zeros(256, 1, CV_32F);
    originalimage.convertTo(originalimage, CV_32F);


    for (int row=0; row< originalimage.rows; row++)
    {
        for (int col= 0; col< originalimage.cols; col++)
        {
            bin= originalimage.at<float>(row, col);
            histogram.at<float>(bin) = histogram.at<float>(bin) + 1;
        }
    }

    return histogram;
}




// Get Cumulative Histogram of an image
Mat Functions::CumulativeHistogram(Mat histogram){

    Mat weighted_histogram= histogram / sum(histogram);         // calculate weighted histogram= (pixels_of_each_bin/total_number_of_pixels)

    // calculate cumulative histogram
    Mat cumulative_histogram = Mat::zeros(weighted_histogram.size(), weighted_histogram.type());
    cumulative_histogram.at<float>(0) = weighted_histogram.at<float>(0);

    for (int i=1; i<256; i++)
    {
        cumulative_histogram.at<float>(i)= weighted_histogram.at<float>(i) + cumulative_histogram.at<float>(i-1);
    }

    return cumulative_histogram;

}





// Draw Histogram of an image
void Functions::DrawHistogram(Mat histogram, string window_name){

    Mat Background(400, 512, CV_8UC3, Scalar(255, 255, 255));
    Mat normalized_histogram;
    normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());

    for (int i = 0; i < 256; i++)
    {
       rectangle(Background, Point(2 * i, Background.rows - normalized_histogram.at<float>(i)),Point(2 * (i + 1), Background.rows), Scalar(255, 0, 0));

    }

    //namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, Background);
}





// Equalize the image (will have higher contrast)
void Functions::Equalize(Mat originalimage){

    Mat histogram= GetHistogram(originalimage);
    Mat weighted_histogram= histogram / sum(histogram);         // calculate weighted histogram= (pixels_of_each_bin/total_number_of_pixels)

    // calculate cumulative histogram
    Mat cumulative_histogram = CumulativeHistogram(histogram);
    cumulative_histogram= cumulative_histogram * 255;


    // Mapping
    Mat GreyValue;
    originalimage.convertTo(GreyValue, CV_32F);                // creates a copy of the original image
    Mat output = Mat::zeros(originalimage.size(), CV_32F);    // the output matrix where final values to be stored

    for (int row=0; row< originalimage.rows; row++)
    {
        for (int col=0; col< originalimage.cols; col++)
        {
            output.at<float>(row, col)= cumulative_histogram.at<float>(GreyValue.at<float>(row, col));
        }
    }



    // Quantization
    output.convertTo(output, CV_8UC1);
    imshow("Equalized Image", output);

}




// Normalize the image
void Functions::Normalize(Mat originalimage, float value){
    Mat normalized;
    originalimage.convertTo(normalized, CV_32F);        // convertTo(OutputArray m, int rtype, double alpha=1, double beta=0)
    normalized= normalized/value;

    imshow("Normalized Image", normalized);
}


