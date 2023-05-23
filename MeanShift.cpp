#include "MeanShift.h"
#include <math.h>

using namespace cv;
using namespace std;

Coordinated::Coordinated(){
    x = -1;
    y = -1;
}



void Coordinated::accumaltePoint(Coordinated Pt){
    x += Pt.x;
    y += Pt.y;
    R += Pt.R;
    G += Pt.G;
    B += Pt.B;
}

void Coordinated::copyPoint(Coordinated Pt){
    x = Pt.x;
    y = Pt.y;
    R = Pt.R;
    G = Pt.G;
    B = Pt.B;
}

// Compute color space distance between two points
float Coordinated::colorDistance(Coordinated Pt){
    return sqrt((R - Pt.R) * (R - Pt.R) + (G - Pt.G) * (G - Pt.G) + (B - Pt.B) * (B - Pt.B));
}

// Compute spatial space distance between two points
float Coordinated::spatialDistance(Coordinated Pt){
    return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

void Coordinated::scalePoint(float scale){
    x *= scale;
    y *= scale;
    R *= scale;
    G *= scale;
    B *= scale;
}

void Coordinated::setPoint(float px, float py, float pR, float pa, float pb){
    x = px;
    y = py;
    R = pR;
    G = pa;
    B = pb;
}

MeanShift::MeanShift(){
}
void MeanShift::MeanShift_Segmentation(Mat& Img, float spatialBW, float colorBW){

//---------------- Mean Shift Filtering -----------------------------
    int ROWS = Img.rows;
    int COLS = Img.cols;
    split(Img, IMGChannels);

    Coordinated curr_point;
    Coordinated prev_point;
    Coordinated PtSum;
    Coordinated Pt;
    int Left, Right, Top, Bottom;
    int num_of_points, step;

    int maxConvergenceSteps = 5;
    float minColorShiftChange = 0.3;
    float minSpatialShiftChange = 0.3;


    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            Left = (j - spatialBW) > 0 ? (j - spatialBW) : 0;
            Right = (j + spatialBW) < COLS ? (j + spatialBW) : COLS;
            Top = (i - spatialBW) > 0 ? (i - spatialBW) : 0;
            Bottom = (i + spatialBW) < ROWS ? (i + spatialBW) : ROWS;

            curr_point.setPoint(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
            step = 0;
            do{
                prev_point.copyPoint(curr_point);
                PtSum.setPoint(0, 0, 0, 0, 0);
                num_of_points = 0;
                for(int x1 = Top; x1 < Bottom; x1++){
                    for(int y1 = Left; y1 < Right; y1++){

                        Pt.setPoint(x1, y1, (float)IMGChannels[0].at<uchar>(x1, y1), (float)IMGChannels[1].at<uchar>(x1, y1), (float)IMGChannels[2].at<uchar>(x1, y1));

                        if(Pt.colorDistance(curr_point) < colorBW){
                            PtSum.accumaltePoint(Pt);
                            num_of_points++;
                        }
                    }
                }
                PtSum.scalePoint(1.0 / num_of_points);
                curr_point.copyPoint(PtSum);
                step++;
            }while((curr_point.colorDistance(prev_point) > minColorShiftChange) && (curr_point.spatialDistance(prev_point) > minSpatialShiftChange) && (step < maxConvergenceSteps));

            Img.at<Vec3b>(i, j) = Vec3b(curr_point.R, curr_point.G, curr_point.B);
        }
    }

    //----------------------- Segmentation ------------------------------
    int RegionNumber = 0;
    int label = -1;
    float *segmented = new float [ROWS * COLS * 3];
    int *MemberModeCount = new int [ROWS * COLS];
    int p[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};

    memset(MemberModeCount, 0, ROWS * COLS * sizeof(int));		// Initialize the MemberModeCount
    split(Img, IMGChannels);

    // Label for each point
    int **Labels = new int *[ROWS];
    for(int i = 0; i < ROWS; i++)
        Labels[i] = new int [COLS];

    // Initialization
    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            Labels[i][j] = -1;
        }
    }

    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS;j ++){
            // If the point is not being labeled
            if(Labels[i][j] < 0){
                Labels[i][j] = ++label;		// Give it a new label number
                // Get the point
                curr_point.setPoint(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));

                // Store each value of Lab
                segmented[label * 3 + 0] = curr_point.R;
                segmented[label * 3 + 1] = curr_point.G;
                segmented[label * 3 + 2] = curr_point.B;

                // Region Growing 8 Neighbours
                vector<Coordinated> NeighbourPoints;
                NeighbourPoints.push_back(curr_point);
                while(!NeighbourPoints.empty()){
                    Pt = NeighbourPoints.back();
                    NeighbourPoints.pop_back();

                    for(int k = 0; k < 8; k++){
                        int x1 = Pt.x + p[k][0];
                        int y1 = Pt.y + p[k][1];
                        if((x1 >= 0) && (y1 >= 0) && (x1 < ROWS) && (y1 < COLS) && (Labels[x1][y1] < 0)){
                            Coordinated P;
                            P.setPoint(x1, y1, (float)IMGChannels[0].at<uchar>(x1, y1), (float)IMGChannels[1].at<uchar>(x1, y1), (float)IMGChannels[2].at<uchar>(x1, y1));

                            // Check the color
                            if(curr_point.colorDistance(P) < colorBW){
                                // Satisfied the color bandwidth
                                Labels[x1][y1] = label;
                                NeighbourPoints.push_back(P);
                                MemberModeCount[label]++;
                                // Sum all color in same region
                                segmented[label * 3 + 0] += P.R;
                                segmented[label * 3 + 1] += P.G;
                                segmented[label * 3 + 2] += P.B;
                            }
                        }
                    }
                }
                MemberModeCount[label]++;
                segmented[label * 3 + 0] /= MemberModeCount[label];		// Get average color
                segmented[label * 3 + 1] /= MemberModeCount[label];
                segmented[label * 3 + 2] /= MemberModeCount[label];
            }
        }
    }
    RegionNumber = label + 1;										// Get region number

    // Get result image from Mode array
    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            label = Labels[i][j];
            float r = segmented[label * 3 + 0];
            float g = segmented[label * 3 + 1];
            float b = segmented[label * 3 + 2];
            Coordinated Pixel;
            Pixel.setPoint(i, j, r, g, b);
            Img.at<Vec3b>(i, j) = Vec3b(Pixel.R, Pixel.G, Pixel.B);
        }
    }


}

