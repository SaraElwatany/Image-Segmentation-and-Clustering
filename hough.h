#ifndef HOUGH_H
#define HOUGH_H


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <map>
#include <random>
#include <time.h>
#include <fstream>


using namespace std;
using namespace cv;

struct Circle{
    double radius;
    int xCenter;
    int yCenter;
    int maxVote;
};


// Any Ellipse can be defined using the center , a(major axis) , b(minor axis) , orientation
struct ellipse_data
{
   vector<double> x0;
   vector<double> y0;
   vector<double> a;
   vector<double> b;
   vector<double> orientation;
};


class Hough
{

public:


    Hough();
//    virtual ~Hough();
        int Line_Transform(unsigned char *image, int width, int height);
        vector<pair<pair<int, int>,pair<int, int>>> GetLines(int threshold);
        const unsigned int* GetAccu(int *w,int*h);
        Mat draw_lines(Mat image,vector<pair<pair<int,int>,pair<int,int>>> lines);
        void HoughVote(Mat &vote,Mat image,double radius,map<int,pair<double,double>>angle,int &maxVote);
        void FindPeak(Mat &Vote, int maxVote, int numberPeaks, vector<Circle> &peakCenters, double radius);
        bool checkCircle(vector<Circle>&Circles,Circle newCircle,int pixel_Interval);
        void Circle_Transform(Mat edge_image, vector<Circle> &Circles, double min_radius, double max_radius);
        Mat drawCircles(Mat image, vector<Circle> Circles, const int limit);
        // Ellipse Detection

        // First Approach
        vector<double> insertAtEnd(vector<double> A, double e);
        int GetIndex(vector<long long unsigned int> vec);
        vector<double> GetBins(double bin_size, double max_vote);
        vector<long long unsigned int> GetHistogram(vector<double>Accumulator, vector<double>bin_edges);
        double Distance(double x1, double y1, double x2, double y2);
        pair<int,int> GetCenter(int x1, int y1, int x2, int y2);
        double MajorAxisLength(int x1, int y1, int x2, int y2);
        double MinorAxisLength(double a, double d, double f);
        double GetFocalLength(double x0, double y0, double x, double y);
        double Orientation(double x1, double y1, double x2, double y2);
        vector<Point> GetEdgePixels(Mat EdgeImg);
        ellipse_data DetectEllipse(Mat EdgeImg, long long unsigned int threshold, float accuracy, int min_size, int max_size);

        // Second Approach
        bool ConfirmEllipse(vector<Point>contour, RotatedRect ellipse);
        Mat Preprocessing(Mat originalImage, Mat GreyImage);


private:
    unsigned int *acc;
    int acc_width;
    int acc_height;
    int image_width;
    int image_height;



};

#endif // HOUGH_H
