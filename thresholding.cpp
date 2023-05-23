#include "thresholding.h"

Thresholding::Thresholding()
{

}
int Thresholding:: optimalThresholding(Mat& image) {
    // Initialize clusters
    vector<int> cluster1 = {image.at<uchar>(0, 0), image.at<uchar>(0, image.cols-1), image.at<uchar>(image.rows-1, 0), image.at<uchar>(image.rows-1, image.cols-1)};
    vector<int> cluster2;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if ((i != 0 && j != 0 && i != image.rows-1 && j != image.cols-1) || (i == 0 && j == 0) || (i == 0 && j == image.cols-1) || (i == image.rows-1 && j == 0) || (i == image.rows-1 && j == image.cols-1))
                continue;
            cluster2.push_back(image.at<uchar>(i, j));
        }
    }

    // Compute initial threshold accorging to the 2 clusters mean
    int threshold = (static_cast<int>(round(mean(cluster1)[0])) + static_cast<int>(round(mean(cluster2)[0]))) / 2;

    // Iterate until convergence
    while (true) {
        // Assign pixels to clusters based on current threshold
        vector<int> newCluster1, newCluster2;
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                if (image.at<uchar>(i, j) <= threshold)
                    newCluster1.push_back(image.at<uchar>(i, j));
                else
                    newCluster2.push_back(image.at<uchar>(i, j));
            }
        }

        // Compute new threshold and check for convergence
        int newThreshold = (static_cast<int>(round(mean(newCluster1)[0])) + static_cast<int>(round(mean(newCluster2)[0]))) / 2;
        if (abs(newThreshold - threshold) < 1e-6)
            break;
        else {
            threshold = newThreshold;
            cluster1 = newCluster1;
            cluster2 = newCluster2;
        }
    }

    // Return threshold value
    return threshold;
}
int Thresholding::otsuThresholding(Mat image)
{
    // initialize histogram array with zeros
    int hist[256] = { 0 };

    // calculate the histogram of the input image
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            // get the intensity value of the pixel at (x,y)
            int intensity = static_cast<int>(image.at<uchar>(y, x));

            // increment the count for that intensity value in the histogram array
            hist[intensity]++;
        }
    }

    // calculate the total number of pixels in the image
    int total = image.rows * image.cols;

    // calculate the sum of intensity values multiplied by their respective counts
    double sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * static_cast<double>(hist[i]);
    }

    // initialize variables for within-class variance, max variance, and threshold value
    double sumB = 0;
    int wB = 0;
    int wF = 0;
    double maxVar = 0;
    int threshold = 0;

    // loop over all possible intensity values to find optimal threshold
    for (int i = 0; i < 256; i++)
    {
        // add the count of the current intensity value to the count of background pixels
        wB += hist[i];
        if (wB == 0) continue;

        // calculate the count of foreground pixels
        wF = total - wB;
        if (wF == 0) break;

        // add the sum of intensity values multiplied by their respective counts to the sum for the background class
        sumB += i * static_cast<double>(hist[i]);

        // calculate the mean intensity values for background and foreground classes
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        // calculate the between-class variance
        double varBetween = static_cast<double>(wB) * static_cast<double>(wF) * (mB - mF) * (mB - mF);

        // update the threshold value if the between-class variance is higher than the current maximum
        if (varBetween > maxVar)
        {
            maxVar = varBetween;
            threshold = i;
        }
    }

    // return the optimal threshold value
    return threshold;
}


Mat Thresholding:: globalThresholding(Mat original,  double thresholdValue , int maxValue)
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
Mat Thresholding:: localThresholding(Mat image, int blockSize, std::string thresholdMethod) {
    Mat result = image.clone();

    // Divide the image into blocks
    for (int i = 0; i < image.rows; i += blockSize) {
        for (int j = 0; j < image.cols; j += blockSize) {
            int blockHeight = std::min(blockSize, image.rows - i);
            int blockWidth = std::min(blockSize, image.cols - j);
            Mat subImage = image(cv::Range(i, i+blockHeight), cv::Range(j, j+blockWidth));

            // Apply thresholding to the block
            int localThreshold;
            if (thresholdMethod == "otsu") {
                localThreshold = otsuThresholding(subImage);
            } else if (thresholdMethod == "optimal") {
                localThreshold = optimalThresholding(subImage);
            } else {
                // Invalid threshold method specified
                return result;
            }
            threshold(subImage, subImage, localThreshold, 255, THRESH_BINARY);

            // Copy the thresholded block back to the result image
            subImage.copyTo(result(cv::Range(i, i+blockHeight), cv::Range(j, j+blockWidth)));
        }
    }

    return result;
}


pair<int, int> Thresholding:: spectral_thresholding(cv::Mat image) {

    // Calculate the histogram of the input image
    Mat histogram;
    int channels[] = {0};
    int histSize[] = {256};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    calcHist(&image, 1, channels, cv::Mat(), histogram, 1, histSize, ranges);


    // Calculate the cumulative sum of the histogram values
    Mat cumulative_sum;
    reduce(histogram, cumulative_sum, 0, cv::REDUCE_SUM);

    // Calculate the total number of pixels in the image
    int total = image.rows * image.cols;

    // set the threshold values
    double tau1 = 0.0;
    double tau2 = 0.0;

    // set to 0.5 to balance the trade-off between the two threshold values
    double alpha = 0.5;

    int background_pixels = 0;
    int foreground_pixels = 0;
    double background_pixels_sum = 0.0;
    double foreground_pixels_sum = 0.0;
    double max_Variance = 0.0;
    int threshold_1 = 0;
    int threshold_2 = 0;

    //iterates over all possible threshold values from 0 to 255
    for (int i = 0; i <= 255; ++i) {

        background_pixels = background_pixels + histogram.at<float>(i);

        if (background_pixels == 0) continue;

        foreground_pixels = total - background_pixels;

        if (foreground_pixels == 0) break;

        background_pixels_sum = background_pixels_sum + i * histogram.at<float>(i);
        foreground_pixels_sum = cumulative_sum.at<float>(0) - background_pixels_sum;

        double background_mean = background_pixels_sum / background_pixels;
        double foreground_mean = foreground_pixels_sum / foreground_pixels;

        //calculates the within-class variance and between-class variance

        double variance_difference = background_pixels * foreground_pixels * (background_mean - foreground_mean) * (background_mean - foreground_mean);

        // updates the threshold values if the between-class variance is the max variance
        if (variance_difference > max_Variance) {
            max_Variance = variance_difference;
            threshold_1 = i;
        }

        tau1 = tau1 + histogram.at<float>(i) * i;

        //updates tau2
        if (tau1 > alpha * total && tau2 == 0) {
            tau2 = i;
        }
    }

    // threshold_2 is the average of threshold_1 and tau2
    threshold_2 = round((threshold_1 + tau2) /2.0 );

    // compensate the threshold values
    threshold_1 = threshold_1 - 50 ;
    threshold_2 = threshold_2 - 50;

    return make_pair(threshold_1, threshold_2);

}


Mat Thresholding::Double_Thresholding (Mat img , pair<int, int> thresholds ){

    // Apply the threshold to the image
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) <= thresholds.second)
                img.at<uchar>(i, j) = 0;
            else if (img.at<uchar>(i, j) > thresholds.second && img.at<uchar>(i, j) <= thresholds.first)
                img.at<uchar>(i, j) = 128;
            else
                img.at<uchar>(i, j) = 255;
        }
    }
    return img;
}



Mat Thresholding:: Global_Spectral(Mat img){

    // Apply double thresholding
    pair<int, int> thresholds = Thresholding::spectral_thresholding(img);

    Mat thresholded_img = Thresholding::Double_Thresholding(img,thresholds);

    return thresholded_img;
}




Mat Thresholding:: Local_Spectral(Mat image,int blockSize ){

    Mat localThreshold;
    for (int i = 0; i < image.rows; i += blockSize) {
        for (int j = 0; j < image.cols; j += blockSize) {
            int blockHeight = std::min(blockSize, image.rows - i);
            int blockWidth = std::min(blockSize, image.cols - j);
            Mat subImage = image(cv::Range(i, i+blockHeight), cv::Range(j, j+blockWidth));

            std::pair<int, int> thresholds = Thresholding:: spectral_thresholding(subImage);

            localThreshold =  Thresholding::Double_Thresholding(subImage,thresholds);

            subImage.copyTo(image(cv::Range(i, i+blockHeight), cv::Range(j, j+blockWidth)));
        }
    }
    return image;
}
