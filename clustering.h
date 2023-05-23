#ifndef CLUSTERING_H
#define CLUSTERING_H


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>


#include <iostream>
#include <ctime>
#include <cmath>



using namespace cv;
using namespace std;



// Struct for each centroid
struct Center{

    double blueValue;
    double greenValue;
    double redValue;

};




class Clustering
{



public:


    Clustering();
    void InitializeCentroids(Mat originalImage, int k, vector<Scalar> &clustersCentroids, vector<vector<Point>> &Clusters);
    double GetEuclideanDistance(Scalar pixel, Scalar Centroid);
    void BuildClusters(Mat originalImage, int k, vector<Scalar> clustersCentroids, vector<vector<Point>> & Clusters);
    double AdjustCentroids(Mat originalImage, int k, vector<Scalar> & clustersCentroids, vector<vector<Point>> Clusters, double & oldCentroid, double newCentroid);
    Mat PaintImage(Mat &segmentImage, int k, vector<vector<Point>> Clusters);
    Mat GetKMeans(Mat InputImage, int k);
    Point GetMinimum(vector<double> vec);
    double GetEuclideanDistance(Mat image, Point pt1, Point pt2);
    Mat BuildHeirarchy(Mat image, int no_clusters);



};


#endif // CLUSTERING_H
