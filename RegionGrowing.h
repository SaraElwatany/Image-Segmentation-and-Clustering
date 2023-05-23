#ifndef REGIONGROWING_H
#define REGIONGROWING_H




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


using namespace std;
using namespace cv;

class RegionGrowing{
public:
    RegionGrowing();
    int gray_diff(Mat img, Point current_point, Point temp_point);
    vector<Point> neighboring_pixels (int p);
    Mat Region_Growing(Mat img, vector<pair<int, int>> seed_set, int thresh, int p) ;


};












#endif // REGIONGROWING_H
