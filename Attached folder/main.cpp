//#include "mainwindow.h"


//#include <QApplication>

#include "MeanShift.h"



#include <iostream>
#include <stack>
#include <cassert>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using std::cout;
using std::endl;
using std::stack;
using std::vector;

using namespace std;
using namespace cv;


cv::Mat BGR_To_LUV(cv::Mat img) {
    int height = img.rows;
    int width = img.cols;
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < height; i++) {
        for (int c = 0; c < width; c++) {
            float b = img.at<cv::Vec3b>(i, c)[0] / 255.0;
            float g = img.at<cv::Vec3b>(i, c)[1] / 255.0;
            float r = img.at<cv::Vec3b>(i, c)[2] / 255.0;

            float x = (0.412453 * r) + (0.35758 * g) + (0.180423 * b);
            float y = (0.212671 * r) + (0.71516 * g) + (0.072169 * b);
            float z = (0.019334 * r) + (0.119193 * g) + (0.950227 * b);

            if (y > 0.008856) {
                output.at<cv::Vec3b>(i, c)[0] = (116.0 * std::pow(y, 1.0/3)) - 16;
            } else {
                output.at<cv::Vec3b>(i, c)[0] = 903.3 * y;
            }

            float u_dash = 4.0 * x / (x + (15.0 * y) + (3.0 * z));
            float v_dash = 9.0 * y / (x + (15.0 * y) + (3.0 * z));

            output.at<cv::Vec3b>(i, c)[1] = 13 * output.at<cv::Vec3b>(i, c)[0] * (u_dash - 0.19793943);
            output.at<cv::Vec3b>(i, c)[2] = 13 * output.at<cv::Vec3b>(i, c)[0] * (v_dash - 0.46831096);

            output.at<cv::Vec3b>(i, c)[0] = ((255.0 / 100) * output.at<cv::Vec3b>(i, c)[0]);
            output.at<cv::Vec3b>(i, c)[1] = ((255.0 / 354) * output.at<cv::Vec3b>(i, c)[1] + 134);
            output.at<cv::Vec3b>(i, c)[2] = ((255.0 / 262) * output.at<cv::Vec3b>(i, c)[2] + 140);
        }
    }

    return output;
}




int gray_diff(Mat img, Point current_point, Point temp_point) {
    return abs(int(img.at<uchar>(current_point)) - int(img.at<uchar>(temp_point)));
}


vector<Point> connects_selection(int p) {
    vector<Point> connects;
    if (p != 0) {
        connects = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
    } else {
        connects = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    }
    return connects;
}



Mat fit(Mat img, vector<pair<int, int>> seed_set, int thresh, int p=1) {
    Mat seed_mark = Mat::zeros(img.size(), CV_32SC1);
    vector<Point> seed_list;
    std::cout << "Seed set: ";
    for (auto& point : seed_set) {
        std::cout << "(" << point.first << ", " << point.second << ") ";
    }
    std::cout << std::endl;

    for (auto seed : seed_set) {
        Point seed_point(seed.first, seed.second);
        seed_list.push_back(seed_point);
    }

    int label = 1;
    vector<Point> connects = connects_selection(p);
    while (!seed_list.empty()) {
        Point current_pixel = seed_list[0];
        seed_list.erase(seed_list.begin());
        seed_mark.at<int>(current_pixel) = label;
        for (auto connect : connects) {
            Point tmp_point(current_pixel.x + connect.x, current_pixel.y + connect.y);
            if (tmp_point.x < 0 || tmp_point.y < 0 || tmp_point.x >= img.rows || tmp_point.y >= img.cols) {
                continue;
            }
            int gray_diff_val = gray_diff(img, current_pixel, tmp_point);
            if (gray_diff_val < thresh && seed_mark.at<int>(tmp_point) == 0) {
                seed_mark.at<int>(tmp_point) = label;
                seed_list.push_back(tmp_point);
            }
        }
    }
    return seed_mark;
}







vector<pair<int, int>> seed_set; // Stores the seeds


void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        seed_set.push_back(pair<int, int>(x, y));
        Mat* p_image = static_cast<Mat*>(userdata);
        cv::Scalar colour( 0, 0, 255 );
        cv::circle( *p_image, Point(x, y), 4, colour, FILLED );
        imshow("CT slice", *p_image);
    }
}


int main() {
    cv::Mat src = cv::imread("D:/Qt__Projects/cvtest3/Capture6.PNG",0);


    Mat colour_ct_slice;
    cvtColor(src, colour_ct_slice, COLOR_GRAY2RGB);

    namedWindow("CT slice", WINDOW_AUTOSIZE); // Create a window
    imshow("CT slice", colour_ct_slice); // Show our image inside the created window
    setMouseCallback("CT slice", mouseCallback, &colour_ct_slice); // Register the callback function

    waitKey(0);

    std::cout << "Seed set: ";
    for (auto& point : seed_set) {
        std::cout << "(" << point.first << ", " << point.second << ") ";
    }
    std::cout << std::endl;
waitKey(0);

    Mat segmented_image = fit(src, seed_set,5);




    namedWindow("Segmentation", WINDOW_AUTOSIZE);
    normalize(segmented_image, segmented_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
     cv::threshold(segmented_image, segmented_image, 128, 255, THRESH_BINARY_INV);// Create a window
    imshow("Segmentation", segmented_image);
    waitKey(0); // Wait for any keystroke in the window



    printf("Mission complete...");
    return 0;
}

