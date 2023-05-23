#include "harris_operator.h"

harris_operator::harris_operator()
{

}


//The nonMaxSuppression function is defined to suppress non-maximum values in the corner response function.
//This function iterates over each pixel and sets the pixel value to zero if it is not a local maximum
// The function takes the response image and the window size as inputs

void harris_operator:: nonMaxSuppression(Mat& response, int window_size)
{
    int r = window_size / 2;
    Mat_<float>::iterator it = response.begin<float>() + r * response.cols + r;
    Mat_<float>::iterator end = response.end<float>() - r * response.cols - r;
    while (it != end)
    {
        float val = *it;
        if (val > 0)
        {
            Mat_<float>::iterator it1 = it - r * response.cols - r;
            Mat_<float>::iterator it2 = it + r * response.cols + r;
            for (; it1 <= it2; it1 += response.cols)
            {
                for (int j = -r; j <= r; ++j, ++it1)
                {
                    if (*it1 > val)
                    {
                        val = 0;
                        break;
                    }
                }
                if (val == 0)
                    break;
            }
            *it = val;
        }
        ++it;
    }
}

Mat harris_operator:: convertToGray(Mat img) {
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = img.clone();
    }
    return gray;
}

//The harrisCorner function is defined to calculate the corner response function for the input grayscale image.

Mat harris_operator:: harrisCorner(Mat gray,int window_size ,double alpha) {
    Mat R(gray.size(), CV_64F);
    Mat Ix, Iy;
    //calculates the x and y derivatives of the image using the Sobel filter
    Sobel(gray, Ix, CV_64F, 1, 0, 3);
    Sobel(gray, Iy, CV_64F, 0, 1, 3);
    Mat Ix2, Iy2, IxIy;
    //calculate the products of the derivatives to get Ix^2, Iy^2, and IxIy.
    Ix2 = Ix.mul(Ix);
    Iy2 = Iy.mul(Iy);
    IxIy = Ix.mul(Iy);
    Mat Ix2_sum, Iy2_sum, IxIy_sum;
    //These products are filtered using a kernel
    Mat kernel(window_size, window_size, CV_64F, 1);
    filter2D(Ix2, Ix2_sum, CV_64F, kernel);
    filter2D(Iy2, Iy2_sum, CV_64F, kernel);
    filter2D(IxIy, IxIy_sum, CV_64F, kernel);
    double A, B, C, d, t;
    // the corner response function is calculated for each pixel using the formula R = det(M) - alpha*trace(M)^2.

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            A = Ix2_sum.at<double>(i, j);
            B = IxIy_sum.at<double>(i, j);
            C = Iy2_sum.at<double>(i, j);
            d = A * C - B * B;
            t = A + C;
            R.at<double>(i, j) = d - alpha * t * t;
        }
    }
    nonMaxSuppression(R, window_size);
    //return the corner response function image
        return R;
}


//The setThreshold function is defined to set the threshold for the corner response function.
void harris_operator::setThreshold(Mat& R,double threshold_level) {
    double R_max;
    minMaxLoc(R, NULL, &R_max, NULL, NULL);
    //calculate the maximum value in the response function image and sets the threshold to be a fraction of that value.
    double threshold = threshold_level * R_max;
    for (int i = 0; i < R.rows; i++) {
        for (int j = 0; j < R.cols; j++) {
            if (R.at<double>(i, j) < threshold) {
                // set all values below the threshold to zero.
                R.at<double>(i, j) = 0;
            }
        }
    }
    return;
}
//The getLocalMax function is defined to get the local maximum values in the corner response function image.

void harris_operator:: getLocalMax(Mat R, Mat& localMax) {
    Mat dilated;
    //The function first dilates the response function image to get the local maximum values.
    dilate(R, dilated, Mat());
    // compare the original response function image to the dilated image to get the local maximum values.
    compare(R, dilated, localMax, CMP_EQ);
    // subtract 255 from the local maximum image and adds the dilated image to get the final local maximum values.
    localMax = localMax - 255 + (dilated > 0);
    return;
}

//The drawPointsToImg function is defined to draw the detected corners on the input image.
//The function takes the input image and the local maximum values image as inputs.

Mat  harris_operator:: drawPointsToImg(Mat img, Mat localMax) {
    localMax.convertTo(localMax, CV_64F);
//It then iterates over each pixel in the local maximum values image and draws a circle on the input image if the pixel value is nonzero
    for (int i = 0; i < localMax.rows; i++) {
        for (int j = 0; j < localMax.cols; j++) {
            if (localMax.at<double>(i, j) != 0) {
                Point p = Point(j, i);
                 circle(img, p, 1, Scalar(0,0,255),1, 1, 0);
            }
        }
    }
    return img;
}
