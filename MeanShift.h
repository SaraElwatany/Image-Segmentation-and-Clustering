#include <iostream>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>



//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;


// 5-Dimensional Point
class Coordinated{

    public:
        float x, y;
        float R, G, B;

        Coordinated();

        void accumaltePoint(Coordinated);
        void copyPoint(Coordinated);
        float colorDistance(Coordinated);
        float spatialDistance(Coordinated);
        void scalePoint(float);
        void setPoint(float, float, float, float, float);
        };

class MeanShift{
    public:
        vector<Mat> IMGChannels;
        MeanShift();
        void MeanShift_Segmentation(Mat& Img, float spatialBW, float colorBW);

};
