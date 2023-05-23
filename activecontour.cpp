#include "activecontour.h"
#include <opencv2/opencv.hpp>

ActiveContour::ActiveContour()
{

}

// This function define the initial contour
std::vector<Point> ActiveContour::initial_contour(cv::Point center, int radius){
    // Create an empty vector of Points to hold the initial contour
    std::vector<Point> initial_snake;
    double current_angle = 0;
    // Set the resolution to 3.6 degrees per point (360 degrees / 100 points)
    double resolution = 360 / 100.0;
    for (int i = 0; i < 100; i++){
        std::vector<double> angle = { current_angle };
        std::vector<double> x = {};
        std::vector<double> y = {};
        // Convert polar coordinates to cartesian coordinates
        cv::polarToCart(radius, angle, x, y, true);
        // Calculate the x and y coordinates of the point on the contour
        int x_point = (int)(x[0] + center.x);
        int y_point = (int)(y[0] + center.y);
        current_angle += resolution;
        initial_snake.push_back(Point(x_point, y_point));
    }
    return initial_snake;
}


// This function calculates the internal energy of a contour point,which is a measure of how much the point deviates
//from the overall shape of the contour
double ActiveContour::  calcInternalEnergy(cv::Point pt, cv::Point prevPt, cv::Point nextPt, double alpha) {
     // Calculate the differences in x and y coordinates between the current point  and its neighboring points
    double dx1 = pt.x - prevPt.x;
    double dy1 = pt.y - prevPt.y;
    double dx2 = nextPt.x - pt.x;
    double dy2 = nextPt.y - pt.y;
    // Calculate the curvature of the point using the formula: (dx1 * dy2 - dx2 * dy1) / (dx1^2 + dy1^2)^(3/2)
    //This curvature value represents the rate at which the direction of the contour changes at this point.
    double curvature = (dx1 * dy2 - dx2 * dy1) / pow(dx1*dx1 + dy1*dy1, 1.5);
     // change the curvature according to alpha value
    return alpha * curvature;
}


// This function calculates the external energy of a contour point,which is a measure of how much the point
//"likes" or "dislikes" being in its current position based on the image content.
double ActiveContour::  calcExternalEnergy(Mat img, cv::Point pt, double beta) {
    //Find the grayscale intensity at the current point's (x, y) and change it  according to beta value.
    return -beta * img.at<uchar>(pt.y, pt.x);
}


// This function calculates the energy associated with a balloon term for active contours.
// The balloon term is used to control the contour's shape by attracting or repelling it from a given point.
double ActiveContour:: calcBalloonEnergy(cv::Point pt, cv::Point prevPt, double gamma) {
// Calculate the change in x and y coordinates between the current and previous points.
    double dx = pt.x - prevPt.x;
    double dy = pt.y - prevPt.y;

// Calculate the energy associated with the balloon term using the distance between the current and previous points.
    return gamma * (dx*dx + dy*dy);
}

// This function is resbonsible for updating the initial contour according to energy functions.
std::vector<Point> ActiveContour::  contourUpdating(Mat inputImg, Mat &outputImg,Point center, int radius,
                                                    int numOfIterations,double alpha, double beta, double gamma)
{
    std::vector<Point> snake_points = initial_contour(center, radius);
    Mat grayImg=inputImg;
//    Convert_To_Gray(inputImg, grayImg);
    blur(grayImg, grayImg, Size(5, 5));

    // Iterate for multiple iterations
    for (int i = 0; i < numOfIterations; i++)
    {
        int numPoints = snake_points.size();
        std::vector<Point> newCurve(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            Point pt = snake_points[i];
            Point prevPt = snake_points[(i - 1 + numPoints) % numPoints];
            Point nextPt = snake_points[(i + 1) % numPoints];
            double minEnergy = DBL_MAX;
            Point newPt = pt;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    Point movePt(pt.x + dx, pt.y + dy);
                    // calculate the total energy
                    double internal_e = calcInternalEnergy(movePt, prevPt, nextPt, alpha);
                    double external_e = calcExternalEnergy(grayImg, movePt, beta);
                    double balloon_e = calcBalloonEnergy(movePt, prevPt, gamma);
                    double energy = internal_e + external_e + balloon_e;
                    if (energy < minEnergy)
                    {
                        minEnergy = energy;
                        newPt = movePt;
                    }
                }
            }
            newCurve[i] = newPt;
        }
         // Update the snake_points for the next iteration
        snake_points = newCurve;
    }
// this part only to draw the final contour on the ouyput image.
    outputImg = grayImg.clone();
    for (int i = 0; i < snake_points.size(); i++)
    {
        circle(outputImg, snake_points[i], 4, Scalar(0, 0, 255), -1);
        if (i > 0)
        {
            line(outputImg, snake_points[i - 1], snake_points[i], Scalar(255, 0, 0), 2);
        }
    }
    line(outputImg, snake_points[0], snake_points[snake_points.size() - 1], Scalar(255, 0, 0), 2);

    return snake_points;
}

// This function take the final contour as a parameter and return vector contatin the direction of movement to each point
std::vector<int> ActiveContour:: calculateChainCode(const std::vector<Point>& contour) {
    std::vector<int> chainCode;
    const int numPoints = contour.size();
    int prevDir = 0;  // initialize the previous direction to zero
    for (int i = 0; i < numPoints; i++) {
        Point currPt = contour[i];
        Point prevPt = contour[(i - 1 + numPoints) % numPoints];
        // calculate the difference in x and y coordinates
        int dx = currPt.x - prevPt.x;
        int dy = currPt.y - prevPt.y;
        // Calculate the direction of the current point relative to the previous point
        int dir = -1;
        // diffrent directions represent the movement in positive and negative x and y directions
        if (dx == 0 && dy == 1) {
            dir = 0;
        } else if (dx == -1 && dy == 1) {
            dir = 1;
        } else if (dx == -1 && dy == 0) {
            dir = 2;
        } else if (dx == -1 && dy == -1) {
            dir = 3;
        } else if (dx == 0 && dy == -1) {
            dir = 4;
        } else if (dx == 1 && dy == -1) {
            dir = 5;
        } else if (dx == 1 && dy == 0) {
            dir = 6;
        } else if (dx == 1 && dy == 1) {
            dir = 7;
        }
         // Adjust the direction of the current point based on the previous direction
        dir = (dir - prevDir + 8) % 8;

        chainCode.push_back(dir);
        // Update the previous direction for the next iteration
        prevDir = dir;
    }
      return chainCode;
 }
