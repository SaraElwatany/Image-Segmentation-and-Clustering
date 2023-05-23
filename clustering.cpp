#include "clustering.h"

Clustering::Clustering()
{

}


/********************************************************************** K-MEANS **************************************************************************/


// Initialize K random centroids and their associated empty vector for pixels
void Clustering::InitializeCentroids(Mat originalImage, int k, vector<Scalar> &clustersCentroids, vector<vector<Point>> &Clusters)
{

    // Initialize random object
    RNG random(cv::getTickCount());

    for(int cluster =0; cluster<k; cluster++){

        // Get random pixel (Centroid)
        Point centroid;
        centroid.x = random.uniform(0, originalImage.cols);
        centroid.y = random.uniform(0, originalImage.rows);
        Scalar centroid_intensity = originalImage.at<Vec3b>(centroid.y, centroid.x);

        // Get intesities of centroid and save it
        Scalar centerK(centroid_intensity.val[0], centroid_intensity.val[1], centroid_intensity.val[2]);
        clustersCentroids.push_back(centerK);

        // Get vector for future associated pixel to each centroid
        vector<Point> ptCluster;
        Clusters.push_back(ptCluster);
    }

}






// Get the Euclidian Distance for pixels' intensities
double Clustering::GetEuclideanDistance(Scalar pixel, Scalar Centroid){

    // Get intensity difference
    double diffBlue = pixel.val[0] - Centroid[0];
    double diffGreen = pixel.val[1] - Centroid[1];
    double diffRed = pixel.val[2] - Centroid[2];

    //Get euclidian distance
    double euclidian_distance = sqrt(pow(diffBlue, 2) + pow(diffGreen,2) + pow(diffRed,2));

    return euclidian_distance;
}








// Compute the pixels for initial clusters
void Clustering::BuildClusters(Mat originalImage, int k, vector<Scalar> clustersCentroids, vector<vector<Point>> & Clusters)
{


    // For each pixel, find closest cluster
    for(int row = 0 ; row<originalImage.rows; row++){
        for(int col = 0; col<originalImage.cols; col++){


            double minDistance = INFINITY;
            int closestClusterIndex = 0;
            Scalar pixel = originalImage.at<Vec3b>(row,col);


            // Iterate through each cluster present
            for(int cluster = 0; cluster<k; cluster++){

                Scalar clusterPixel = clustersCentroids[cluster];   // Get the intensity of the associated Centroid of current cluster

                // Get Euclidean Distance between the centoid & current pixel
                double distance = GetEuclideanDistance(pixel, clusterPixel);

                // Update to closest cluster center
                if(distance < minDistance){

                    minDistance = distance;
                    closestClusterIndex = cluster;

                }
            }

            // Push pixel into associated cluster
            Clusters[closestClusterIndex].push_back(Point(col,row));

        }
    }

}








// Compute the new Centroids for each cluster
double Clustering::AdjustCentroids(Mat originalImage, int k, vector<Scalar> & clustersCentroids, vector<vector<Point>> Clusters, double & oldCentroid, double newCentroid){

    double diffChange;

    // Adjust cluster centroid to mean of associated cluster's pixels
    for(int cluster =0; cluster<k; cluster++){

        vector<Point> ptInCluster = Clusters[cluster];
        double newBlue = 0;
        double newGreen = 0;
        double newRed = 0;

        // Compute mean intensity values for the cluster for the associated 3 channels
        for(int i=0; i<ptInCluster.size(); i++){

            Scalar pixel = originalImage.at<Vec3b>(ptInCluster[i].y,ptInCluster[i].x);
            newBlue += pixel.val[0];
            newGreen += pixel.val[1];
            newRed += pixel.val[2];

        }

        newBlue /= ptInCluster.size();
        newGreen /= ptInCluster.size();
        newRed /= ptInCluster.size();

        //Assign new intensity to the cluster's centroid
        Scalar newPixel(newBlue, newGreen, newRed);
        clustersCentroids[cluster] = newPixel;

        //Compute distance between the old and new values
        newCentroid += GetEuclideanDistance(newPixel,clustersCentroids[cluster]);

    }


    newCentroid /= k;

    //get difference between previous iteration change
    diffChange = abs(oldCentroid - newCentroid);
    cout << "diffChange is: " << diffChange << endl;
    oldCentroid = newCentroid;

    return diffChange;

}







// Get the segmented image
Mat Clustering::PaintImage(Mat &segmentImage, int k, vector<vector<Point>> Clusters){

    srand(time(NULL));

    // Assign random color to each cluster
    for(int cluster=0; cluster<k; cluster++){

        vector<Point> ptInCluster = Clusters[cluster];
        // Get random colour for each cluster
        Scalar randomColor(rand() % 255,rand() % 255,rand() % 255);

        // For each pixel in cluster change color to fit cluster
        for(int indx=0; indx<ptInCluster.size(); indx++){

            Scalar pixelColor = segmentImage.at<Vec3b>(ptInCluster[indx]);
            pixelColor = randomColor;

            segmentImage.at<Vec3b>(ptInCluster[indx])[0] = pixelColor.val[0];
            segmentImage.at<Vec3b>(ptInCluster[indx])[1] = pixelColor.val[1];
            segmentImage.at<Vec3b>(ptInCluster[indx])[2] = pixelColor.val[2];
        }
    }

    return segmentImage;
}





// Perform K-Means Clustering
Mat Clustering::GetKMeans(Mat InputImage, int k)
{

    // Set up cluster centroids, cluster vector, and threshold to terminate the iterations
    vector<Scalar> clustersCentroids;
    vector< vector<Point> > ptInClusters;
    double threshold = 0.1;      // Stop whenever the difference in intensities between new & old centroid is equal or below the threshold
    double oldCenter=INFINITY;
    double newCenter=0;
    double diffChange = oldCenter - newCenter;

    // Constraint on K (whenever it is below 1 set to k)
    if (k<1){

        k =1;
    }


    // Initialize K random centroids
    InitializeCentroids(InputImage, k, clustersCentroids, ptInClusters);

    // Iterate until cluster centers nearly stop moving (using threshold)
    while(diffChange > threshold){

        newCenter = 0;        // Reset change

        // Clear associated pixels for each cluster
        for(int cluster=0; cluster<k; cluster++){
             ptInClusters[cluster].clear();
        }

        //find all closest pixel to cluster centers
        BuildClusters(InputImage, k, clustersCentroids, ptInClusters);
        // Recompute cluster centers values
        diffChange = AdjustCentroids(InputImage, k, clustersCentroids, ptInClusters, oldCenter, newCenter);
    }


    // Show output image
    Mat OutputImage = InputImage.clone();
    OutputImage = PaintImage(OutputImage, k, ptInClusters);
    //imshow("Segmentation", OutputImage);
    //waitKey(0);

    return OutputImage;
}




/********************************************************************** Agglomerative **************************************************************************/





// Get the minimum non-zero element in a vector & its index
Point Clustering::GetMinimum(vector<double> vec) {

    double min=0;
    int minIndex=0;
    int flag=0;

    for(int i = 0; i< int(vec.size()); i++) {

        double min_spare = vec[i];

        if(min_spare > 0 && (flag==0)){

            min= min_spare;
            minIndex= i;
            flag+=1;
        }

        if((min_spare < min) && (min_spare > 0)) {

            min = min_spare;
            minIndex = i;

        }


    }

    return Point(min, minIndex);
}








// Get Euclidean Distance Between 2 points in the (R, G, B) 3D plane
double Clustering::GetEuclideanDistance(Mat image, Point pt1, Point pt2){

    Mat image_double;
    image.convertTo(image_double, CV_64F, 1.0/255);

    vector<Mat> bgr_planes;
    split(image_double, bgr_planes);

    double p1, p2;
    double Eucl_Dist = 0.0;


    for(int i=0; i<3; i++){

    p1 = bgr_planes[i].at<double>(pt1.x, pt1.y);
    p2 = bgr_planes[i].at<double>(pt2.x, pt2.y);

    Eucl_Dist = Eucl_Dist + ((p2-p1) * (p2-p1));

    }

    return sqrt(Eucl_Dist);
}







// Perform Agglomerative Clustering & Display the image
Mat Clustering::BuildHeirarchy(Mat image, int no_clusters){


    int rows = image.rows;
    int cols = image.cols;
    int no_pixels = rows*cols;          // Number of the available pixels
    int no_pixels_temp= no_pixels;      // Temporary variable to hold the updated number of pixels (Clusters)


    // Vector Containing all of the image's pixels (Flatten the image)
    vector<Point> pixels;
    for(int row1=0; row1<rows; row1++){
        for(int col1=0; col1<cols; col1++){

            pixels.push_back(Point (row1, col1));

        }
    }



    // Vector Containing all of the available initial clusters positions (initially each pixel represents a cluster)
    vector<vector<Point>> clusters_pixels(no_pixels);
    for(int pixel=0; pixel<no_pixels; pixel++){

        Point pt= pixels[pixel];

        clusters_pixels[pixel].push_back(pt);

    }



    // Vector Containing all of the available euclidean distances for each cluster
    vector<vector<double>> clusters_eucl_dist(no_pixels);
    for(int pixel=0; pixel<no_pixels; pixel++){

        Point pt1= pixels[pixel];
        for(int pixel2=0; pixel2<=pixel; pixel2++){

            Point pt2= pixels[pixel2];
            double euc_distance_temp = GetEuclideanDistance(image, pt1, pt2);

            clusters_eucl_dist[pixel].push_back(euc_distance_temp);

        }

    }





  // Loop till you reach the required number of clusters
  for(int cluster= no_pixels; cluster > no_clusters; cluster--){


    int roww, coll;
    double m=0.0;

    for(int pixel=0; pixel<no_pixels_temp; pixel++){

        double min_row_temp= GetMinimum(clusters_eucl_dist[pixel]).x;
        int min_col_indx= GetMinimum(clusters_eucl_dist[pixel]).y;

        if(pixel==0){

            m=min_row_temp;

        }

        else if(min_row_temp < m){

            m=min_row_temp;
            roww= pixel;
            coll= min_col_indx;
        }

    }



    // Update the clusters
    for(int point=0; point < int(clusters_pixels[coll].size()); point++){

        Point pt2= clusters_pixels[coll][point];
        clusters_pixels[roww].push_back(pt2);

    }


    for(int pixel=coll; pixel<no_pixels; pixel++){

        clusters_pixels[pixel]= clusters_pixels[pixel+1];

    }

    clusters_pixels[no_pixels_temp-1].pop_back();
    no_pixels_temp-=1;




    // Transfer the euclidean values of the upper cluster(which will be combined with another one corresponding to min ED found) to a temporary vector
    int limit= int(clusters_eucl_dist[roww].size());
    vector<double> clusters_eucl_dist_temp;
    for(int i=0; i < limit; i++){

        double val= clusters_eucl_dist[roww][i];
        clusters_eucl_dist_temp.push_back(val);

    }


    // Clear the row containing euclidean distances of our destined newly formed cluster
    for(int i=0; i < limit; i++){

        clusters_eucl_dist[roww].pop_back();

    }


    // Update the euclidean row for each pixel
    for(int i=0; i < limit; i++){

        double val1 = clusters_eucl_dist_temp[i];
        double val2 = clusters_eucl_dist[coll][i];
        double mean = (val1+val2) / 2;

        clusters_eucl_dist[roww].push_back(mean);

        if(i ==(limit-1))
            clusters_eucl_dist[roww][i]=0;

    }


    // Shift the rows up & columns to the left after the update
    for(int pixel=coll; pixel<no_pixels; pixel++){

        clusters_eucl_dist[pixel]= clusters_eucl_dist[pixel+1];

    }

    clusters_eucl_dist[no_pixels_temp].pop_back();

  }



  /*********************** Paint Clusters on the Image **********************/

  Mat Segmented_image;
  Segmented_image = image.clone();

  //Loop through each cluster
  for (int cluster=0; cluster< int(clusters_pixels.size()); cluster++) {

      // Create a random color for the cluster
      Scalar color;
      int rand_num= rand() % 256;
      color= Scalar(rand_num, rand_num, rand_num);
     // Draw each point in the cluster using the specified color
     for (int point=0; point< int(clusters_pixels[cluster].size()); point++) {

         Point pt = clusters_pixels[cluster][point];
         circle(Segmented_image, pt, 2, color, FILLED, LINE_AA);

     }

  }

  return Segmented_image;
  //imshow("HOO!!", Segmented_image);

}
