#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>


using namespace cv;
using namespace std;
using namespace std::chrono;


// keypoint class
class keypoint
{

    public:
    keypoint(int oct = 0, int lyr = 0, Point p = Point(0, 0), double scl = 0, float ang = 0):
        octave(oct), layer(lyr), scale(scl), pt(p), angle(ang) {}

    static void SwapToKeyPoint(vector<keypoint>& kpts, vector<KeyPoint>& KPts);

    public:
    int octave;
    int layer;
    double scale;
    float angle;
    float response;
    int oct;
    Point pt;
};



// Build Pyramid class
class pyramid
{

public:
    void Append(int oct, Mat& img);
    void Build(int oct) { pyr.resize(oct); }
    void Clear() { pyr.clear(); }
    int Octaves() { return pyr.size(); }
    vector<Mat>& operator[] (int oct);

private:
    vector<vector<Mat>>pyr;
};



// Scale Invariante Transform Class
class Sift
{

public:
    Sift();
    Sift(int s, double sigma = 1.6) : S(s), Sigma(sigma), debug(0) { Layers = s + 2; K = pow(2., 1. / s); }
    bool Detect_KeyPoints(const Mat& img, vector<keypoint>& kpts, Mat& fvec);
    Mat Draw_Blob(Mat image,int between_lyr);
//    void Get_time(auto start, auto stop);


private:
    void GetPyramid();
    void ExtractFeatures(vector<keypoint>& kpts);
    bool KeyPoints_Localization(keypoint& kpt);
    void DominantOrientations(keypoint& kpt, vector<float>& angs);
    void FeatureDescribtors(vector<keypoint>& kpts, Mat& Des_vec);

public:
    int debug;

private:
    int Octaves;        // Octaves ->  0 1 2 3
    int Layers;         // Layers = 6 - 1 = 5
    double Sigma;
    int S;              // S = 3
    double K;

    Mat img_org;
    Mat img_blur;

    pyramid Gauss_pyr;
    pyramid DOG_pyr;

};


