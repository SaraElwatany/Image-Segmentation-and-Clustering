#include "hough.h"


//**********************************************************************Line************************************************************************

Hough::Hough():acc(0),acc_width(0),acc_height(0),image_width(0),image_height(0)
{
}

//Hough::~Hough()
//{
//    if(acc)
//        free(acc);
//}


int Hough::Line_Transform (unsigned char*image,int w,int h){

image_width=w;
image_height=h;

//create accumulator
double hough_space=((sqrt(2.0)*double((w>h)?w:h))/2.0);
acc_width=180;
acc_height=(hough_space*2.0);

acc=(unsigned int*)calloc(acc_width*acc_height,sizeof(unsigned int));

double center_x=w/2;
double center_y=h/2;


//calculate the hough transform
for(int y=0;y<h;y++){
    for(int x=0;x<w;x++){
        if(image[y*w+x]>0){
            for(int t=0;t<acc_width;t++){
                double d = ( ((double)x - center_x) * cos((double)t *CV_PI/180.0 )) + (((double)y - center_y) * sin((double)t * CV_PI/180.0));
                acc[ (int)((round(d + hough_space) * 180.0)) + t]++;
            }
        }
    }

}
return 0;
}


    vector<pair<pair<int, int>, pair<int, int> > > Hough::GetLines(int threshold){
    vector<pair<pair<int,int>,pair<int,int>>> lines;
    if(acc==0)
        return lines;

    //find local maxima
    for(int r=0;r<acc_height;r++){
        for(int t=0;t<acc_width;t++){
            if(acc[r*acc_width+t]>threshold){
                    int max = acc[(r*acc_width) + t];
                for(int ly=-4;ly<=4;ly++){
                    for(int lx=-4;lx<=4;lx++){
                        if(r+ly>=0 && r+ly<acc_height && t+lx>=0 && t+lx<acc_width){
                            if((int)acc[(r+ly)*acc_width+(t+lx)]>max){
                                max=acc[(r+ly)*acc_width+(t+lx)];
                                ly=lx=5;
                            }
                        }
                    }
                }
                if(max>(int)acc[(r*acc_width)+t])
                    continue;

                    int x1,y1,x2,y2;
                    x1=y1=x2=y2=0;
                    //convert to cartesian
                    if(t>45 &&t<=135){
                        x1=0;
                        y1=((double)(r-(acc_height/2))-((x1-(image_width/2))*cos(t*CV_PI/180.0)))/sin(t*CV_PI/180.0)+(image_height/2);
                        x2=image_width-0;
                        y2=((double)(r-(acc_height/2))-((x2-(image_width/2))*cos(t*CV_PI/180.0)))/sin(t*CV_PI/180.0)+(image_height/2);
                    }
                    else{
                        y1=0;
                        x1=((double)(r-(acc_height/2))-((y1-(image_height/2))*sin(t*CV_PI/180.0)))/cos(t*CV_PI/180.0)+(image_width/2);
                        y2=image_height-0;
                        x2=((double)(r-(acc_height/2))-((y2-(image_height/2))*sin(t*CV_PI/180.0)))/cos(t*CV_PI/180.0)+(image_width/2);
                    }
                    lines.push_back(pair<pair<int,int>,pair<int,int>>(pair<int,int>(x1,y1),pair<int,int>(x2,y2)));
                }
            }
        }

    return lines;
}



const unsigned int* Hough::GetAccu(int *w,int *h){
   *w=acc_width;
    *h=acc_height;
    return acc;
}


Mat Hough::draw_lines(Mat image,vector<pair<pair<int,int>,pair<int,int>>>lines){
 Mat output_image;
  cvtColor(image, output_image, COLOR_GRAY2BGR);
//draw lines
vector<pair<pair<int, int>,pair<int, int> > >::iterator it;
    for(it=lines.begin();it!=lines.end();it++){
    line(output_image,Point(it->first.first,it->first.second),Point(it->second.first,it->second.second),Scalar(0,0,255),2,8);
}
return output_image;
}







//**********************************************************************Circle************************************************************************

void Hough::HoughVote(Mat &vote,Mat image,double radius,map<int,pair<double,double>>angle,int &maxVote){
    int rows=image.rows;
    int cols=image.cols;
    Scalar pixel_image;
    int a,b;

    //loop through each pixel of image
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            pixel_image=image.at<uchar>(i,j);
            if(pixel_image.val[0]>0){
                //loop through each theta
                for(int theta=0;theta<360;theta++){
                    a=(int)(i-radius*angle[theta].first);
                    b=(int)(j-radius*angle[theta].second);

                    //only increase vote if value are inbounds
                    if(a>=0 && a<rows && b>=0 && b<cols){
                        vote.at<short>(a,b)++;
                        if(vote.at<short>(a,b)>maxVote){
                            maxVote=vote.at<short>(a,b);
                        }
                    }
                }

            }
        }
    }

}

void Hough::FindPeak(Mat &Vote, int maxVote, int numberPeaks, vector<Circle> &peakCenters, double radius){

    int threshold = 0.8 * maxVote;

    //If threshold under 100, it's probably not a circle
    if(threshold < 100) threshold = 100;

    Point max_Point;
    double maxValue;
    Circle new_Circle;

    int peak_numbers = 0;
    int neighborhood = 4;

    //loop until desired nnumber of peaks are reach
    while(peak_numbers < numberPeaks){

        //find max value of HoughVote and location a/b
        minMaxLoc(Vote, NULL, &maxValue, NULL, &max_Point);

        //if maxValue over threshold
        if(maxValue > threshold){
            peak_numbers++;

            //create new Circle
            new_Circle.maxVote = maxValue;
            new_Circle.xCenter = max_Point.x;
            new_Circle.yCenter = max_Point.y;
            new_Circle.radius = radius;

            //store newCircle
            peakCenters.push_back(new_Circle);

            //set neighborhood zone to zero to avoid circle in same region
            for(int i=max_Point.x-neighborhood; i<=max_Point.x+neighborhood; i++){
                for(int j=max_Point.y-neighborhood; j<max_Point.y+neighborhood; j++){
                    Vote.at<short>(j,i)=0;
                }
            }
        }
        else{
            break;
        }
    }
}

bool Hough::checkCircle(vector<Circle>&Circles,Circle newCircle,int pixel_Interval){
    bool isthere=false;
    for(vector<Circle>::iterator it=Circles.begin();it!=Circles.end();){
        if((newCircle.xCenter>=it->xCenter-pixel_Interval && newCircle.xCenter<=it->xCenter+pixel_Interval) && (newCircle.yCenter>=it->yCenter-pixel_Interval && newCircle.yCenter<=it->yCenter+pixel_Interval)){
           if(it->maxVote<newCircle.maxVote){
               it=Circles.erase(it);
               isthere=false;
               break;
           }
           else{
            if(it->radius*2<newCircle.radius){
                isthere=false;
                break;
            }
            else{
                isthere=true;
                ++it;

           }
        }
    }
    else{
        ++it;
    }
    }
//Only circle returned false will be added to the BestCircle vector

    return isthere;
}


void Hough::Circle_Transform(Mat edge_image, vector<Circle> &Circles, double min_radius, double max_radius){

    int rows = edge_image.rows;
    int cols = edge_image.cols;
    int maxVote = 0;
    int numberPeaks = 10;
    int pixelInterval = 15;
    int size[2] = {rows, cols};
    Mat hough_vote;
    vector<Circle> peakCenters;

    //Compute all possible theta from degree to radian and store them into a map to avoid overcomputation
    map<int, pair<double, double>> thetaMap;
    for (int thetaD=0; thetaD <360; thetaD++) {
        double thetaR = static_cast<double>(thetaD) * (CV_PI / 180);
        thetaMap[thetaD].first = cos(thetaR);
        thetaMap[thetaD].second = sin(thetaR);
    }

    //Loop for each possible radius - radius range may need to be changed following the image (size of circle)
    for (double r = min_radius; r <max_radius; r+=1.0){

        //Initialize maxVote, accumulator, peakCenters to zeros
        maxVote= 0;
        hough_vote = Mat::zeros(2,size, CV_16U);
        peakCenters.clear();

        //Compute Vote for each edge pixel
        HoughVote(hough_vote, edge_image, r, thetaMap, maxVote);

        //Find Circles with maximum votes
        FindPeak(hough_vote, maxVote, numberPeaks, peakCenters, r);

        //For each Circle find, only keep the best ones (max votes) and remove duplicates
        for(int i=0; i<peakCenters.size(); i++){
            bool isthere = checkCircle(Circles, peakCenters[i], pixelInterval);
            if(!isthere){
                Circles.push_back(peakCenters[i]);
            }
        }
    }

}

Mat Hough::drawCircles(Mat image, vector<Circle> Circles, const int limit)
{
    Mat result;

    //Transform image to RGB to draw circles in color
    cvtColor(image, result, COLOR_GRAY2BGR);

    for (int i=0; i < limit; ++i) {
        circle(result, Point(Circles[i].xCenter, Circles[i].yCenter), Circles[i].radius, Scalar(255,0,0),4);
    }
    return result;
}




//**********************************************************************Ellipse************************************************************************





// First Approach

// Function to append in a Given Vector
vector<double> Hough::insertAtEnd(vector<double> A, double e)
{
   A.push_back( e );
   return A;
}





// Get position of the max element in vector
int Hough::GetIndex(vector<long long unsigned int>vec)
{

    int index;
    // Get the iterator pointing to largest element in vector
    auto it = std::max_element(vec.begin(), vec.end());
    if(it != vec.end())
    {
        // Get the distance of iterator from the beginning of vector
        index = std::distance(vec.begin(), it);
    }

    return index;
}






// Create the required bins
vector<double> Hough::GetBins(double bin_size, double max_vote)
{

    int size = int(max_vote);
    vector<double>bins(size);
    int index = 0;

    for (double bin=0; bin < max_vote + bin_size; bin+=bin_size){

        bins[index] = double(bin * bin_size);
        index+=1;
    }

    return bins;
}






// Calculate the values at histogram bins
vector<long long unsigned int> Hough::GetHistogram(vector<double> Accumulator1, vector<double> bin_edges)
{

    const size_t no_edges = bin_edges.size();
    const size_t acc_size = Accumulator1.size();
    const size_t size =  no_edges - 1;

    vector<long long unsigned int> histogram(size);

    for (size_t bin_edge = 0; bin_edge < no_edges; bin_edge++)
    {
        for (size_t vote = 0; vote < acc_size; vote++)
        {
            if ((bin_edges[bin_edge] < Accumulator1[vote]) && (Accumulator1[vote] < bin_edges[bin_edge + 1]))
            {
                histogram[bin_edge] = histogram[bin_edge] + 1;
            }
        }
    }

    return histogram;
}






// Get the center of an ellipse using any 2 random points
pair<int,int> Hough::GetCenter(int x1, int y1, int x2, int y2)
{
    return make_pair(cvRound((x1 + x2)*0.5), cvRound((y1 + y2)*0.5));
}






// Get the distance between 2 pixels (points)
double Hough::Distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}





// Get the square of half the length of the major axis
double Hough::MajorAxisLength(int x1, int y1, int x2, int y2)
{
    return Distance(x1, y1, x2, y2) * 0.5;
}





// Get the square of the length of the minor axis
double Hough::MinorAxisLength(double a, double d, double f)
{

    double b_sqrd;
    double cos_tau_sqrd = ( (a * a) + (d * d) - (f) ) / (2 * a * d);
    cos_tau_sqrd *= cos_tau_sqrd;
    double sin_tau_sqrd = 1 - cos_tau_sqrd;
    double denimonator = ( (a * a) - ( (d * d) * (cos_tau_sqrd) ) );

    if (denimonator > 0 && cos_tau_sqrd < 1){

        b_sqrd = ( (a * a) * (d * d) * sin_tau_sqrd ) / denimonator ;

    }

    return b_sqrd;
}






double Hough::GetFocalLength(double x1, double y1, double x, double y)
{

    double dx = x - x1;
    double dy = y - y1;
    double f = (dx * dx) + (dy * dy);

    return f;
}






// Get orientation of an ellipse
double Hough::Orientation(double x1, double y1, double x2, double y2)
{
    if (x1 == x2)
        return CV_PI * 0.5;
    return atan((y2 - y1) / (x2 - x1));
}






// Get cooerdinates of non zero pixels
vector<Point>Hough::GetEdgePixels(Mat EdgeImg)
{

    vector<Point>EdgePixels;

    // Loop through each array pixel (x1,y1)
    for (int x=0; x < EdgeImg.rows; x++)
    {
        for (int y=0; y < EdgeImg.cols; y++)
        {

            if (EdgeImg.at<uchar>(x,y) != 0){

                Point edge_point;
                edge_point.x =  x;
                edge_point.y = y;
                EdgePixels.push_back(Point(edge_point));

            }


        }
    }

    return EdgePixels;
}









// Return found ellipses
ellipse_data Hough::DetectEllipse(Mat EdgeImg, long long unsigned int threshold, float accuracy, int min_size, int max_size)
{

    vector<Point>EdgePixels = GetEdgePixels(EdgeImg);
    ellipse_data ellipse;
    vector<double>Accumulator;
    int bin_size = accuracy * accuracy;
    double max_b_sqrd;          // Max allowed range for minor axis


    // If the user doesn't give the maximum acceptable range for the minor axis it is set to the mimimum number of the image's dimension
    if (max_size == NULL){

        if (EdgeImg.rows < EdgeImg.cols){
            max_b_sqrd = round(0.5 * EdgeImg.rows);
        }

        else{

            max_b_sqrd = round(0.5 * EdgeImg.cols);
        }


        max_b_sqrd *= max_b_sqrd;
    }

    else{
         max_b_sqrd = max_size * max_size;
        }



    // Loop through each edge pixel (x1,y1)
    for (int pixel1=0; pixel1 < EdgePixels.size(); pixel1++)
    {


        // Loop through each edge one more time (x2,y2)
        for (int pixel2=0; pixel2 < pixel1; pixel2++)
        {


            int x1 = EdgePixels[pixel1].x;
            int y1 = EdgePixels[pixel1].y;
            int x2 = EdgePixels[pixel2].x;
            int y2 = EdgePixels[pixel2].y;

            // Compute half the length of the major axis of the potential ellipse
            double a = MajorAxisLength(x1, y1, x2, y2);

            // Check the distance between first (x1,y1) & second (x2,y2) pixel
            if (min_size * 0.5 < a && a < 200){


                // Get the center of the potential ellipse
                pair<int,int>center;
                center = GetCenter(x1, y1, x2, y2);
                int x0 = center.first;
                int y0 = center.second;


                // Loop through each edge for the third time (x,y)
                for (int pixel3=0; pixel3 < EdgePixels.size(); pixel3++)
                {


                    int x = EdgePixels[pixel3].x;
                    int y = EdgePixels[pixel3].y;
                    double d = Distance(x, y, x0, y0);
                    double f = GetFocalLength(x1, y1, x, y);

                    if (d > min_size){

                        double b_sqrd =  MinorAxisLength(a, d, f);
                        // b_sqrd is limited to avoid histogram memory overflow
                        if (b_sqrd <= max_b_sqrd){

                            Accumulator = insertAtEnd(Accumulator , b_sqrd);

                        }

                    }

                }




                if(Accumulator.size() > 0){

                    vector<double>bins;
                    double maxs = *max_element(Accumulator.begin(), Accumulator.end());
                    bins = GetBins(bin_size, maxs);
                    vector<long long unsigned int>histogram;
                    histogram = GetHistogram(Accumulator, bins);
                    long long unsigned int histogram_max = *max_element(histogram.begin(), histogram.end());    //20


                    if (histogram_max >= threshold){

                        double orientation = Orientation(x1, y1, x2, y2);         // Compute the orientation of the potential ellipse
                        int max_index = GetIndex(histogram);
                        double b = sqrt(bins[max_index]);
                        double b_spare = b;

                        if (orientation != 0){

                            orientation = M_PI - orientation;
                            if (orientation > M_PI) {

                                orientation = orientation - (M_PI / 2.);
                                b = a;
                                a = b_spare;

                            }



                        ellipse.x0.push_back(x0);
                        ellipse.y0.push_back(y0);
                        ellipse.a.push_back(a);
                        ellipse.b.push_back(b);
                        ellipse.orientation.push_back(orientation);

                        }

                    }




                }   // End of condition on Accumulator length

                Accumulator.clear(); // clear accumulator array

                }       // End of condition on major axis (a)

            }

        }

    return ellipse;
}







// Second approach



// Confirm that the assumed ellipse matches the contour
bool Hough::ConfirmEllipse(vector<Point>contour, RotatedRect BoundEllipse)
{

    vector<Point>AssumedPoints;         //Store all the points detected on the assumed ellipse
    //get the center, major, minor axes as well as the orientation of the assumed ellipse
    Point2f center = BoundEllipse.center;
    double a_sqrd = pow(BoundEllipse.size.width*0.5,2);      // major axis squared
    double b_sqrd = pow(BoundEllipse.size.height*0.5,2);     // minor axis squared
    double orientation = (BoundEllipse.angle*3.1415926)/180; // orientation of an ellipse


    //the upper half part of the ellipse
    for(int i=0; i < BoundEllipse.size.width; i++)
    {

        double x = - BoundEllipse.size.width * 0.5 + i;
        double y_left = sqrt( (1 - (x*x/a_sqrd)) * b_sqrd );

        cv::Point2f rotate_point_left;
        rotate_point_left.x =  cos(orientation) * x - sin(orientation) * y_left;
        rotate_point_left.y = + sin(orientation) * x + cos(orientation) * y_left;

        rotate_point_left += center;        //translate the point
        AssumedPoints.push_back(Point(rotate_point_left));
    }


    //the bottom half part of ellipse
    for(int i=0;i<BoundEllipse.size.width;i++)
    {
        double x = (BoundEllipse.size.width * 0.5) - i;
        double y_right = - sqrt((1 - (x*x/a_sqrd)) * b_sqrd);

        cv::Point2f rotate_point_right;
        rotate_point_right.x =  cos(orientation) * x - sin(orientation) * y_right;
        rotate_point_right.y = + sin(orientation) * x + cos(orientation) * y_right;

        rotate_point_right += center;       //translate the point

        AssumedPoints.push_back(Point(rotate_point_right));
    }


    vector<vector<Point>>contours;
    contours.push_back(AssumedPoints);
    double a0 = matchShapes(AssumedPoints, contour, cv::CONTOURS_MATCH_I1, 0);  // Get the amount of disimilarity between the contour and our assumption
    if (a0>0.01)
    {
        return false;
    }

    return true;
}





// Preprocess the image of interest
Mat Hough::Preprocessing(Mat originalImage, Mat GreyImage)
{

    Mat threshold_output;       // Initializing Matrix to store the thresholded image
    vector<vector<Point>>contours;    // Initializing vector to store the detected contours
    int threshold_value = threshold(GreyImage, threshold_output, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    findContours(threshold_output, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);    // find contours in a given image

    //fit ellipse
    vector<RotatedRect>BoundEllipse(contours.size()); // Initializing vector to store the rectangular shape around each contour
    for(int contour = 0; contour < contours.size(); contour++)
    {
        //point size check
        if(contours[contour].size()<10)
        {
            continue;
        }
        //point area
        if(contourArea(contours[contour])<10)
        {
            continue;
        }
        BoundEllipse[contour] = fitEllipse(Mat(contours[contour]));       // Bound each contour with rectangular shape
        //  Check whether the assumed ellipse matches the given contour, otherwise continue
        if(!ConfirmEllipse(contours[contour],BoundEllipse[contour]))
        {
            continue;
        }
        ellipse(originalImage, BoundEllipse[contour], (0, 0, 255), 2);  // Drawing Candidate ellipses on original image
    }


    return originalImage;
}
