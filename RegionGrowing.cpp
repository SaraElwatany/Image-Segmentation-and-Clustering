#include "RegionGrowing.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "mainwindow.h"



using namespace std;
using namespace cv;


RegionGrowing::RegionGrowing()
{

}


int RegionGrowing::gray_diff(Mat img, Point current_point, Point temp_point){
    return abs(int(img.at<uchar>(current_point)) - int(img.at<uchar>(temp_point)));
}


vector<Point> RegionGrowing ::neighboring_pixels(int p){
    vector<Point> connects;
    if (p != 0) {
        connects = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
    } else {
        connects = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    }
    return connects;
}


Mat RegionGrowing::Region_Growing(Mat img, vector<pair<int, int>> seed_set, int thresh, int p){
    Mat seed_mark = Mat::zeros(img.size(), CV_32SC1);
    vector<Point> seed_list;
    for (auto seed : seed_set) {
        Point seed_point(seed.first, seed.second);
        seed_list.push_back(seed_point);
    }
    int label = 1;
    vector<Point> connected = neighboring_pixels(p);
    while (!seed_list.empty()) {
        Point current_pixel = seed_list[0];
        seed_list.erase(seed_list.begin());
        seed_mark.at<int>(current_pixel) = label;
        for (auto connect : connected) {
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








