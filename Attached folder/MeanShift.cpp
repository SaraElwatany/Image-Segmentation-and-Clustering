#include "MeanShift.h"
#include <math.h>

using namespace cv;
using namespace std;

Point5D::Point5D(){
    x = -1;
    y = -1;
}

Point5D::~Point5D(){
}

void Point5D::accumaltePoint(Point5D Pt){
    x += Pt.x;
    y += Pt.y;
    R += Pt.R;
    G += Pt.G;
    B += Pt.B;
}

void Point5D::copyPoint(Point5D Pt){
    x = Pt.x;
    y = Pt.y;
    R = Pt.R;
    G = Pt.G;
    B = Pt.B;
}

// Compute color space distance between two points
float Point5D::colorDistance(Point5D Pt){
    return sqrt((R - Pt.R) * (R - Pt.R) + (G - Pt.G) * (G - Pt.G) + (B - Pt.B) * (B - Pt.B));
}

// Compute spatial space distance between two points
float Point5D::spatialDistance(Point5D Pt){
    return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

void Point5D::scalePoint(float scale){
    x *= scale;
    y *= scale;
    R *= scale;
    G *= scale;
    B *= scale;
}

void Point5D::setPoint(float px, float py, float pR, float pa, float pb){
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
    // Same as MSFiltering function
    int ROWS = Img.rows;
    int COLS = Img.cols;
    split(Img, IMGChannels);

    Point5D PtCur;
    Point5D PtPrev;
    Point5D PtSum;
    Point5D Pt;
    int Left, Right, Top, Bottom;
    int NumPts, step;

    int maxConvergenceSteps = 5;
    float minColorShiftChange = 0.3;
    float minSpatialShiftChange = 0.3;


    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            Left = (j - spatialBW) > 0 ? (j - spatialBW) : 0;
            Right = (j + spatialBW) < COLS ? (j + spatialBW) : COLS;
            Top = (i - spatialBW) > 0 ? (i - spatialBW) : 0;
            Bottom = (i + spatialBW) < ROWS ? (i + spatialBW) : ROWS;

            PtCur.setPoint(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
            step = 0;
            do{
                PtPrev.copyPoint(PtCur);
                PtSum.setPoint(0, 0, 0, 0, 0);
                NumPts = 0;
                for(int x1 = Top; x1 < Bottom; x1++){
                    for(int y1 = Left; y1 < Right; y1++){

                        Pt.setPoint(x1, y1, (float)IMGChannels[0].at<uchar>(x1, y1), (float)IMGChannels[1].at<uchar>(x1, y1), (float)IMGChannels[2].at<uchar>(x1, y1));

                        if(Pt.colorDistance(PtCur) < colorBW){
                            PtSum.accumaltePoint(Pt);
                            NumPts++;
                        }
                    }
                }
                PtSum.scalePoint(1.0 / NumPts);
                PtCur.copyPoint(PtSum);
                step++;
            }while((PtCur.colorDistance(PtPrev) > minColorShiftChange) && (PtCur.spatialDistance(PtPrev) > minSpatialShiftChange) && (step < maxConvergenceSteps));

            Img.at<Vec3b>(i, j) = Vec3b(PtCur.R, PtCur.G, PtCur.B);
        }
    }

    //----------------------- Segmentation ------------------------------
    int RegionNumber = 0;			// Reigon number
    int label = -1;					// Label number
    float *Mode = new float [ROWS * COLS * 3];					// Store the Lab color of each region
    int *MemberModeCount = new int [ROWS * COLS];				// Store the number of each region

    int dxdy[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};	// Region Growing


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
                PtCur.setPoint(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));

                // Store each value of Lab
                Mode[label * 3 + 0] = PtCur.R;
                Mode[label * 3 + 1] = PtCur.G;
                Mode[label * 3 + 2] = PtCur.B;

                // Region Growing 8 Neighbours
                vector<Point5D> NeighbourPoints;
                NeighbourPoints.push_back(PtCur);
                while(!NeighbourPoints.empty()){
                    Pt = NeighbourPoints.back();
                    NeighbourPoints.pop_back();

                    // Get 8 neighbours
                    for(int k = 0; k < 8; k++){
                        int x1 = Pt.x + dxdy[k][0];
                        int y1 = Pt.y + dxdy[k][1];
                        if((x1 >= 0) && (y1 >= 0) && (x1 < ROWS) && (y1 < COLS) && (Labels[x1][y1] < 0)){
                            Point5D P;
                            P.setPoint(x1, y1, (float)IMGChannels[0].at<uchar>(x1, y1), (float)IMGChannels[1].at<uchar>(x1, y1), (float)IMGChannels[2].at<uchar>(x1, y1));

                            // Check the color
                            if(PtCur.colorDistance(P) < colorBW){
                                // Satisfied the color bandwidth
                                Labels[x1][y1] = label;				// Give the same label
                                NeighbourPoints.push_back(P);		// Push it into stack
                                MemberModeCount[label]++;			// This region number plus one
                                // Sum all color in same region
                                Mode[label * 3 + 0] += P.R;
                                Mode[label * 3 + 1] += P.G;
                                Mode[label * 3 + 2] += P.B;
                            }
                        }
                    }
                }
                MemberModeCount[label]++;							// Count the point itself
                Mode[label * 3 + 0] /= MemberModeCount[label];		// Get average color
                Mode[label * 3 + 1] /= MemberModeCount[label];
                Mode[label * 3 + 2] /= MemberModeCount[label];
            }
        }
    }
    RegionNumber = label + 1;										// Get region number

    // Get result image from Mode array
    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            label = Labels[i][j];
            float r = Mode[label * 3 + 0];
            float g = Mode[label * 3 + 1];
            float b = Mode[label * 3 + 2];
            Point5D Pixel;
            Pixel.setPoint(i, j, r, g, b);
            Img.at<Vec3b>(i, j) = Vec3b(Pixel.R, Pixel.G, Pixel.B);
        }
    }

    //--------------- Delete Memory Applied Before -----------------------
    delete[] Mode;
    delete[] MemberModeCount;

    for(int i = 0; i < ROWS; i++){
        delete[] Labels[i];
    }
    delete[] Labels;
}


