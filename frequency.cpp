#include "frequency.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>


using namespace cv;
using namespace std;



Frequency::Frequency()
{

}



// Calculate the values at histogram bins
Mat Frequency::GetHistogram(Mat originalimage)
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
Mat Frequency::CumulativeHistogram(Mat histogram){

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
Mat Frequency::DrawHistogram(Mat histogram){

    Mat Background(400, 512, CV_8UC3, Scalar(255, 255, 255));
    Mat normalized_histogram;
    normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());

    for (int i = 0; i < 256; i++)
    {

       rectangle(Background, Point(2 * i, Background.rows - normalized_histogram.at<float>(i)),Point(2 * (i + 1), Background.rows), Scalar(255, 0, 0));

    }

    return Background;
}





// Equalize the image (will have higher contrast)
Mat Frequency::Equalize(Mat originalimage){

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

    return output;

}






// Get the minimum and maximum intensity in a given image
uchar Frequency::getMinMax(Mat image){

   // initializing an array where the minimum intensity lies in the first index and the maximum at the second index
   uchar min_max[2] = {image.at<uchar>(0, 0) , image.at<uchar>(0, 0)};

   for(int row= 0; row < image.rows; row++){
       for(int col= 0; col < image.cols; col++){

           if(min_max[0] < image.at<uchar>(row, col))
              min_max[0] = image.at<uchar>(row, col);

           if(min_max[1] > image.at<uchar>(row, col))
              min_max[1] = image.at<uchar>(row, col);
       }

   }

   return min_max[2];
}







// Normalize the image
Mat Frequency::Normalize(Mat originalimage){

    Mat normalized= originalimage.clone();
    uchar min_max[2]= {getMinMax(originalimage)};
    uchar min = min_max[0] ,  max = min_max[1];
    uchar value2= max-min;

    normalized= (normalized-min)*255 / value2;

    return normalized;
}





// Greyscale of an image
Mat Frequency::GreyScale(Mat originalimage){
    Mat result(originalimage.rows, originalimage.cols, CV_8U);
    for(int i=0;i<originalimage.rows;i++){
      for(int j=0;j<originalimage.cols;j++){
         Vec3b color = originalimage.at<Vec3b>(i,j);
         uchar  gray = (color[0] + color[1] + color[2])/3;
         result.at<uchar>(i,j)=gray;
      }
   }
        return result;
     }




// Split colored image into its 3 channels or get greyscale
Mat Frequency::ImageSplit(Mat originalimage,int type)
{

    Mat result(originalimage.rows, originalimage.cols, CV_8U);
    vector<Mat> bgr_planes;
    split(originalimage, bgr_planes);
    if(type==0){
          result=bgr_planes[0];
     }
    else if(type==1){
          result=bgr_planes[1];
     }
    else if(type==2){
          result=bgr_planes[2];
     }
    else if(type==3){
         result= GreyScale(originalimage);
         }

    return result;

}



