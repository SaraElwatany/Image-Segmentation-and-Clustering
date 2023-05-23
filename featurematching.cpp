#include "featurematching.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <chrono>



using namespace std;
using namespace cv;
using namespace std::chrono;


FeatureMatching::FeatureMatching()
{

}


double FeatureMatching:: sum_of_squared_differences(const std::vector<double>& desc_image1, const std::vector<double>& desc_image2) {
    double ssd = 0.0;
     if (desc_image1.size() != desc_image2.size()) {
    //      Return -1 to indicate error (features have different sizes)
         return -1;
     }
    for (int i = 0; i < desc_image1.size(); i++) {
        ssd+= pow((desc_image1[i] - desc_image2[i]), 2);
    }
double final_ssd = sqrt(ssd);
    return final_ssd;
}

vector<DMatch> FeatureMatching:: match_features(Mat descriptor1, Mat descriptor2, double threshold) {
int KeyPoints1=descriptor1.rows;
int KeyPoints2=descriptor2.rows;
    auto start = high_resolution_clock::now();


    vector<DMatch> matches;
    for (int kp1 = 0; kp1 < KeyPoints1; kp1++) {
        double best_ssd = std::numeric_limits<double>::max();
        int best_index = -1;
        for (int Kp2 = 0; Kp2 < KeyPoints2; Kp2++) {
            double ssd = sum_of_squared_differences(descriptor1.row(kp1), descriptor2.row(Kp2));
            if (ssd < best_ssd) {
                best_ssd = ssd;
                best_index = Kp2;
            }
        }
        if (best_ssd <= threshold) {
            DMatch feature;
            // The index of the feature in the first image
            feature.queryIdx = kp1;
            // The index of the feature in the second image
            feature.trainIdx = best_index;
            // The distance between the two features
            feature.distance = best_ssd;
            matches.push_back(feature);

    }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // Print the duration in seconds
    cout << "Time taken by the algorithm: " << duration.count() / 1000000.0 << " seconds" << endl;


    return matches;
}
Mat FeatureMatching:: element_wise_multiply(Mat a, Mat b) {

    Mat result(a.rows, a.cols, a.type());

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.at<float>(i, j) = a.at<float>(i, j) * b.at<float>(i, j);
        }
    }
    return result;
}

//calculate NCC score between two descriptors.
float FeatureMatching::calculate_NCC(Mat img1_descriptors, Mat img2_descriptors, int i, int j)
{
    Scalar mean1, mean2, stdDev1, stdDev2;
       double numerator = 0.0;
       double denominator1 = 0.0;
       double denominator2 = 0.0;
       for (int k = 0; k < img1_descriptors.cols; ++k) {
           double pixel1 = img1_descriptors.at<float>(i, k);
           double pixel2 = img2_descriptors.at<float>(j, k);
           numerator += (pixel1 - mean1.val[0]) * (pixel2 - mean2.val[0]);
           denominator1 += pow(pixel1 - mean1.val[0], 2);
           denominator2 += pow(pixel2 - mean2.val[0], 2);
       }
       double NCC = numerator / sqrt(denominator1 * denominator2);

       return NCC;
}




//perform feature matching between the two sets of descriptors using NCC.
vector<DMatch>FeatureMatching:: feature_matching_temp(Mat descriptor1, Mat descriptor2, String method) {

    vector<DMatch> matches;
        for (int i = 0; i < descriptor1.rows; ++i) {
            double bestNCC = -1.0;
            int bestIndex = -1;
            for (int j = 0; j < descriptor2.rows; ++j) {

                double NCC = calculate_NCC(descriptor1,  descriptor2, i,j) ;

                if (NCC > bestNCC) {
                    bestNCC = NCC;
                    bestIndex = j;
                }
            }
            if (bestIndex >= 0) {
                DMatch match(i, bestIndex, bestNCC);
                matches.push_back(match);
            }
        }
        return matches;

}

