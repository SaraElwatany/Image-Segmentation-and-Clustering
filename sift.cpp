#include "sift.h"


// Get keypoints
void keypoint::SwapToKeyPoint(vector<keypoint>& kpts, vector<KeyPoint>& KPts)
{

    for (auto kpt : kpts)
    {
        KeyPoint KPt;
        KPt.pt.x = kpt.pt.y * pow(2.f, float(kpt.octave - 1));
        KPt.pt.y = kpt.pt.x * pow(2.f, float(kpt.octave - 1));
        KPt.size = kpt.scale * pow(2.f, float(kpt.octave - 1)) * 2;
        KPt.angle = 360 - kpt.angle;
        KPt.response = kpt.response;
        KPt.octave = kpt.oct;
        KPts.push_back(KPt);
    }


}





// Build Pyramid for Both Gaussian & DOG Images
void pyramid::Append(int oct, Mat& img)
{
    Mat temp;
    img.copyTo(temp);
    pyr[oct].push_back(temp);
}






vector<Mat>& pyramid::operator[](int oct)
{
    return pyr[oct];
}



// Default Constructor
Sift::Sift()
{

}


// *************************************************************** Detect KeyPoints ***************************************************************
bool Sift::Detect_KeyPoints(const Mat& image, vector<keypoint>& kpts, Mat& fvec)
{


    image.copyTo(img_org);  // Copy original image
    // Get no of Octaves
    Octaves = round(log( float(min(img_org.cols, img_org.rows)) ) / log(2.f) - 2.f);

    // Convert image to grayscale
    Mat img_gray, img_gray_float;
    if (img_org.channels() == 3 || img_org.channels() == 4)
        cvtColor(img_org, img_gray, COLOR_BGR2GRAY);
    else
        img_org.copyTo(img_gray);
    img_gray.convertTo(img_gray_float, CV_32FC1);


    // Resize the image
    resize(img_gray_float, img_blur, Size(img_gray_float.cols * 2, img_gray_float.rows * 2), 0, 0, INTER_LINEAR);
    // Perform Gaussian Blurring
    double sigma_init = sqrt(max(Sigma * Sigma - 0.5 * 0.5 * 4, 0.01));
    GaussianBlur(img_blur, img_blur, Size(2 * cvCeil(2 * sigma_init) + 1, 2 * cvCeil(2 * sigma_init) + 1), sigma_init, sigma_init);
    // Get DoG Pyramid
    GetPyramid();
    // Get Keypoints
    ExtractFeatures(kpts);
    // Get Feature Describtors
    FeatureDescribtors(kpts, fvec);

    return true;
}








// ************************************************************** Build Gaussian & DOG Pyramids ***************************************************************
void Sift::GetPyramid()
{
    // Sigma vector to hold the values for each octave
    vector<double> sigma_layers(Layers + 1);
    sigma_layers[0] = Sigma;

    // Compute sigma for each layer present in each octave
    for (int lyr_i = 1; lyr_i < Layers + 1; lyr_i++) // 0 1 2 3 4 5
    {
        double sigma_prev = pow(K, lyr_i - 1) * Sigma;
        double sigma_curr = K * sigma_prev;
        sigma_layers[lyr_i] = sqrt(sigma_curr*sigma_curr - sigma_prev*sigma_prev);
    }


    // Gaussian & DOG vectors
    Gauss_pyr.Clear();
    DOG_pyr.Clear();


    // Initialize empty matrices and vectors to use for storage of Gaussian & DOG Images
    Mat img_gauss, img_DoG;
    img_blur.copyTo(img_gauss);
    Gauss_pyr.Build(Octaves);
    DOG_pyr.Build(Octaves);


    // Loop through each octave
    for (int octave_indx = 0; octave_indx < Octaves; octave_indx++)
    {

        Gauss_pyr.Append(octave_indx, img_gauss);       // Append Image to each octave

        // Loop through each layer in each octave
        for (int lyr_indx = 1; lyr_indx < Layers + 1; lyr_indx++) // 0 1 2 3 4 5
        {
            GaussianBlur(img_gauss, img_gauss, Size(2 * cvCeil(2 * sigma_layers[lyr_indx]) + 1, 2 * cvCeil(2 * sigma_layers[lyr_indx]) + 1), sigma_layers[lyr_indx], sigma_layers[lyr_indx]);
            Gauss_pyr.Append(octave_indx, img_gauss);
            subtract(img_gauss, Gauss_pyr[octave_indx][lyr_indx-1], img_DoG, noArray(), CV_32FC1);     // Get Difference of Gaussian
            DOG_pyr.Append(octave_indx, img_DoG);   // Append DOG

        }

        // Resize the Base Image for next Octave
        resize(Gauss_pyr[octave_indx][Layers - 2], img_gauss, Size(img_gauss.cols / 2, img_gauss.rows / 2), 0, 0, INTER_NEAREST);
    }

}








// *************************************************************** Get KeyPoints ***************************************************************
void Sift::ExtractFeatures(vector<keypoint>& kpts)
{


    kpts.clear();
    int octave_indx, lyr_indx, row, col, pr, pc, kpt_i, ang_i, k;
    float pxs[27];                   // 26 Neighbouring Pixels
    vector<keypoint> kpts_temp;     // KeyPoints Temporary Vector


    int threshold = cvFloor(0.5 * 0.04 / S * 255);    // Contrast Threshold

    // Loop through each Octave
    for (octave_indx = 0; octave_indx < Octaves; octave_indx++)
    {

        // Loop through each layer present in each Octave
        for (lyr_indx = 1; lyr_indx < Layers - 1; lyr_indx++) // 0 1 2 3 4
        {

            const Mat& img_curr = DOG_pyr[octave_indx][lyr_indx];
            // Loop through each row in the image present in the layer
            for (row = 1; row < img_curr.rows - 1; row++)
            {

                // Loop through each column in the image present in the layer
                for (col = 1; col < img_curr.cols - 1; col++)
                {

                    // Get Current image (present_Octave, present_Layer) & its corresponding neighbours from other layers
                    const Mat& prev = DOG_pyr[octave_indx][lyr_indx-1];
                    const Mat& curr = DOG_pyr[octave_indx][lyr_indx];
                    const Mat& next = DOG_pyr[octave_indx][lyr_indx+1];
                    float val = curr.at<float>(row, col);        // Get Current Pixel


                    if (abs(val) >= threshold)
                    {

                        // Fill neighbouring Pixels values in the array
                        for (pr = -1, k = 0; pr < 2; pr++)
                            for (pc = -1; pc < 2; pc++)
                            {
                                pxs[k] = prev.at<float>(row + pr, col + pc);
                                pxs[k + 1] = curr.at<float>(row + pr, col + pc);
                                pxs[k + 2] = next.at<float>(row + pr, col + pc);
                                k += 3;
                            }


                        // Compare each pixel with its 26 neighbours to get maxima or minima
                        if ((val >= pxs[0] && val >= pxs[1] && val >= pxs[2] && val >= pxs[3] && val >= pxs[4] &&
                                val >= pxs[5] && val >= pxs[6] && val >= pxs[7] && val >= pxs[8] && val >= pxs[9] &&
                                val >= pxs[10] && val >= pxs[11] && val >= pxs[12] && val >= pxs[14] && val >= pxs[15] &&
                                val >= pxs[16] && val >= pxs[17] && val >= pxs[18] && val >= pxs[19] && val >= pxs[20] &&
                                val >= pxs[21] && val >= pxs[22] && val >= pxs[23] && val >= pxs[24] && val >= pxs[25] &&
                                val >= pxs[26])

                                ||

                                (val <= pxs[0] && val <= pxs[1]  && val <= pxs[2]  && val <= pxs[3]  && val <= pxs[4] &&
                                val <= pxs[5]  && val <= pxs[6]  && val <= pxs[7]  && val <= pxs[8]  && val <= pxs[9] &&
                                val <= pxs[10] && val <= pxs[11] && val <= pxs[12] && val <= pxs[14] && val <= pxs[15] &&
                                val <= pxs[16] && val <= pxs[17] && val <= pxs[18] && val <= pxs[19] && val <= pxs[20] &&
                                val <= pxs[21] && val <= pxs[22] && val <= pxs[23] && val <= pxs[24] && val <= pxs[25] &&
                                val <= pxs[26]))

                        {

                            keypoint kpt(octave_indx, lyr_indx, Point(row, col));
                            kpts_temp.push_back(kpt);     // Potential Keypoint , Push back in our temporary vector

                        }

                    }

                }
            }

        }
    }


    // Localize Potential KeyPoints by determining whether they are edges or not
    for (kpt_i = 0; kpt_i < kpts_temp.size(); kpt_i++)
    {
        // Detected as an edge -> Continue Loop (skip this pixel/loop)
        if (!KeyPoints_Localization(kpts_temp[kpt_i]))
            continue;

        //***************** Get Orientation for KeyPoint *********************
        vector<float> angs;
        DominantOrientations(kpts_temp[kpt_i], angs);

        for (ang_i = 0; ang_i < angs.size(); ang_i++)
        {
            kpts_temp[kpt_i].angle = angs[ang_i];
            kpts.push_back(kpts_temp[kpt_i]);
        }
    }


}










// ********************************************************** Localize Keypoints (Determine contrast & Edge points) ***************************************************************
// Return true, whenever it is considered keypoint (not an edge & contrast above 0.04)

bool Sift::KeyPoints_Localization(keypoint& kpt)
{
    // Initialize thresholds
    float biasThreshold = 0.5f;
    float contrastThreshold = 0.04f;
    float edgeThreshold = 10.f;

    float normalize_scl = 1.f / 255;  // Normalize Scale 0-255 -> 0-1

    // Bool Parameter to decide whether to proceed to edge condition or exit the loop early, hence avoid overcomputation
    bool isdrop = true;

    // Get information for the Keypoint to check
    int kpt_row = kpt.pt.x;
    int kpt_col = kpt.pt.y;
    int kpt_lyr = kpt.layer;
    // Initialize Hessian Matrix Parameters
    Vec3f dD;
    float dxx, dyy, dss, dxy, dxs, dys;
    //
    Vec3f x_hat;

    // Loop for 5 iterations
    for (int try_i = 0; try_i < 5; try_i++)
    {
        // Get DOG Matrix for each layer & its neighbouring DOGs
        const Mat& DoG_prev = DOG_pyr[kpt.octave][kpt_lyr - 1];
        const Mat& DoG_curr = DOG_pyr[kpt.octave][kpt_lyr];
        const Mat& DoG_next = DOG_pyr[kpt.octave][kpt_lyr + 1];

        // *************************** Compute Hessian Matrix's Parameters *************************************
        dD = Vec3f((DoG_curr.at<float>(kpt_row, kpt_col + 1) - DoG_curr.at<float>(kpt_row, kpt_col - 1)) * normalize_scl * 0.5f,
                (DoG_curr.at<float>(kpt_row + 1, kpt_col) - DoG_curr.at<float>(kpt_row - 1, kpt_col)) * normalize_scl * 0.5f,
                (DoG_prev.at<float>(kpt_row, kpt_col) - DoG_next.at<float>(kpt_row, kpt_col)) * normalize_scl * 0.5f);

        // First Derivative
        dxx = (DoG_curr.at<float>(kpt_row, kpt_col + 1) + DoG_curr.at<float>(kpt_row, kpt_col - 1) - 2 * DoG_curr.at<float>(kpt_row, kpt_col)) * normalize_scl;
        dyy = (DoG_curr.at<float>(kpt_row + 1, kpt_col) + DoG_curr.at<float>(kpt_row - 1, kpt_col) - 2 * DoG_curr.at<float>(kpt_row, kpt_col)) * normalize_scl;
        dss = (DoG_next.at<float>(kpt_row, kpt_col) + DoG_prev.at<float>(kpt_row, kpt_col) - 2 * DoG_curr.at<float>(kpt_row, kpt_col)) * normalize_scl;


        // Second Derivative
        dxy = (DoG_curr.at<float>(kpt_row + 1, kpt_col + 1) - DoG_curr.at<float>(kpt_row + 1, kpt_col - 1)
                    - DoG_curr.at<float>(kpt_row - 1, kpt_col + 1) + DoG_curr.at<float>(kpt_row - 1, kpt_col - 1)) * normalize_scl * 0.25f;
        dxs = (DoG_next.at<float>(kpt_row, kpt_col + 1) - DoG_next.at<float>(kpt_row, kpt_col - 1)
                    - DoG_prev.at<float>(kpt_row, kpt_col + 1) + DoG_prev.at<float>(kpt_row, kpt_col - 1)) * normalize_scl * 0.25f;
        dys = (DoG_next.at<float>(kpt_row + 1, kpt_col) - DoG_next.at<float>(kpt_row - 1, kpt_col)
                    - DoG_prev.at<float>(kpt_row + 1, kpt_col) + DoG_prev.at<float>(kpt_row - 1, kpt_col)) * normalize_scl * 0.25f;

        // Form Hessian Matrix
        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);


        // ************* Compute Parameters *************

        // Solve System of Linear Equations
        x_hat = Vec3f(H.solve(dD, DECOMP_LU));
        for (int x = 0; x < 3; x++)
            x_hat[x] *= -1;

        kpt_col += round(x_hat[0]);
        kpt_row += round(x_hat[1]);
        kpt_lyr += round(x_hat[2]);

        // sigma
        if (abs(x_hat[0]) < biasThreshold && abs(x_hat[1]) < biasThreshold && abs(x_hat[2]) < biasThreshold)  //0.5
        {
            isdrop = false;
            break;
        }


        if (kpt_row < 1 || kpt_row > DoG_curr.rows - 2 ||
            kpt_col < 1 || kpt_col > DoG_curr.cols - 2 ||
            kpt_lyr < 1 || kpt_lyr > Layers - 2)

        {
            break;
        }

    }


    // Unaccepted KeyPoint -> return false
    if (isdrop)
        return false;


    //  Get Current image (Current_Octave, Current_Layer)
    const Mat& DoG_curr = DOG_pyr[kpt.octave][kpt_lyr];

    float D_hat = DoG_curr.at<float>(kpt_row, kpt_col) * normalize_scl + dD.dot(x_hat) * 0.5f;      // Get Current Pixel Value (contrast)
    // Below accepted Contrast threshold
    if (abs(D_hat) * S < contrastThreshold)
        return false;


    // Compute Determinante & Trace of Hessian Matrix
    float trH = dxx + dyy;
    float detH = dxx * dyy - dxy * dxy;

    // Detected as Edge (exit function)
    if (detH <= 0 || trH*trH * edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1) * detH)
        return false;


    // Candidate KeyPoint -> Compute Parameters
    kpt.pt = Point(kpt_row, kpt_col);
    kpt.layer = kpt_lyr;
    kpt.scale = pow(K, kpt_lyr) * Sigma;
    kpt.response = D_hat;
    kpt.oct = kpt.octave + (kpt.layer << 8) + (cvRound((x_hat[2] + 0.5) * 255) << 16);

    return true;

}









// **************************************************************** Get Orientation Of Keypoints ***************************************************************

void Sift::DominantOrientations(keypoint& kpt, vector<float>& angs)

{



    // Get Information for the keypoint
    int kpt_oct = kpt.octave;
    int kpt_lyr = kpt.layer;
    int kpt_row = kpt.pt.x;
    int kpt_col = kpt.pt.y;
    double kpt_scl = kpt.scale;

    int radius = round(3 * 1.5 * kpt_scl);             // radius of drawn window around the keypoint
    const Mat& pyr_G_i = Gauss_pyr[kpt_oct][kpt_lyr];      // Matrix for current image


    int px_row, px_col;                   // (img_r, img_c)

    float temp_hist[36+4] = { 0 };           // Temporary Histogram array for the orientation calculated for each pixel  0 1 2~37 38 39


    // Loop through each point present within the area of chosen window size -> (radius*2+1)*(radius*2+1)
    for (int i = -radius; i <= radius; i++)     // Loop through each row
    {

        px_row = kpt_row + i;
        if (px_row <= 0 || px_row >= pyr_G_i.rows - 1)    // check for pixel range
            continue;


        for (int j = -radius; j <= radius; j++)     // Loop through each column
        {


            px_col = kpt_col + j;
            if (px_col <= 0 || px_col >= pyr_G_i.cols - 1)  // check for pixel range
                continue;

            // Compute difference in X & Y direction to be used in Magnitude & Orientation calculations
            float dx = pyr_G_i.at<float>(px_row, px_col + 1) - pyr_G_i.at<float>(px_row, px_col - 1);
            float dy = pyr_G_i.at<float>(px_row - 1, px_col) - pyr_G_i.at<float>(px_row + 1, px_col);

            // Get Magnitude & Direction (Orientation) for each pixel
            float mag = sqrt(dx * dx + dy * dy);
            float ang = fastAtan2(dy, dx);

            // Get Bin Number
            int bin = round(ang * 36.f / 360.f);

            if (bin >= 36) // Check if the bin is within the range -> Orientation: (0~360) per 10 width -> bins: 0~35
                bin -= 36;
            else if (bin < 0)
                bin += 36;

            // Get the weight for each bin
            float w_G = exp( -(i * i + j * j) / (2 * (1.5 * kpt_scl) * (1.5 * kpt_scl)) );

            // Compute the frequency for the bin * weight    0  1  2~37  38  39
            temp_hist[bin + 2] += mag * w_G;

        }
    }

    // *************************************** Compute the Dominant Orientation ***************************************

    // Histogram Initialization (36 Bins)
    float hist[36] = { 0 };
    temp_hist[0] = temp_hist[36];
    temp_hist[1] = temp_hist[37];
    temp_hist[38] = temp_hist[2];
    temp_hist[39] = temp_hist[3];


    float hist_max = 0;           // Variable to hold the maximum value in the computed histogram
    for (int k = 2; k < 40 - 2; k++)   // Transfer the values present in the temporary histogram to the previous declared array & Get MAX VALUE
    {
        hist[k - 2] = (temp_hist[k - 2] + temp_hist[k + 2]) * (1.f / 16) +
                      (temp_hist[k - 1] + temp_hist[k + 1]) * (4.f / 16) +
                       temp_hist[k] * (6.f / 16);

        if (hist[k - 2] > hist_max)
            hist_max = hist[k - 2];
    }

    // Get Threshold to the Orientation
    float histThreshold = 0.8f * hist_max;

    // ******************** Get Dominant Orientation ********************

    for (int b = 0; b < 36; b++)        // Get the Angles corresponding to the dominant orientation by looping through the histogram
    {

        int kl = b > 0 ? b - 1 : 36 - 1;
        int kr = b < 36 - 1 ? b + 1 : 0;

        if (hist[b] > hist[kl] && hist[b] > hist[kr] && hist[b] >= histThreshold)
        {

            float bin = b + 0.5f * (hist[kl] - hist[kr]) / (hist[kl] - 2 * hist[b] + hist[kr]);     // Get the bin (0 ~ 35)

            if (bin < 0)                // bin
                bin += 36;
            else if (bin >= 36)
                bin -= 36;

            // Get the Angle corresponding to the candidate BIN
            float ang = bin * (360.f / 36);
            if (abs(ang - 360.f) < FLT_EPSILON)
                ang = 0.f;

            angs.push_back(ang);    // Store the VALUE OF DOMINANT ORIENTATIONS
        }
    }

}








// *************************************************************** Feature Descriptors ***************************************************************

void Sift::FeatureDescribtors(vector<keypoint>& kpts, Mat& Des_vec)

{


    Des_vec.create(kpts.size(), 128, CV_32FC1);    // Descriptor Vector for the Dominant Orientations around the Keypoint


    for (int kpt_i = 0; kpt_i < kpts.size(); kpt_i++)
    {

        // Initializing Variables for rows, columns
        int r1, r2, k, ri, ci, oi;

        // Get Information for each keypoint
        int kpt_oct = kpts[kpt_i].octave;
        int kpt_lyr = kpts[kpt_i].layer;
        int kpt_r = kpts[kpt_i].pt.x;
        int kpt_c = kpts[kpt_i].pt.y;
        double kpt_scl = kpts[kpt_i].scale;
        float kpt_ang = kpts[kpt_i].angle;

        int d = 4;      // Window Width OR Circle Radius (4 * 4)
        int n = 8;      // Number of Bins present in the histogram


        const Mat& Gauss_pyr_i = Gauss_pyr[kpt_oct][kpt_lyr];    // Matrix of Current Image

        float hist_width = 3 * kpt_scl;     // Width of histogram bin


        // Get the new  range of angles of pixels
        float cos_t = cosf(kpt_ang * CV_PI / 180) / hist_width;
        float sin_t = sinf(kpt_ang * CV_PI / 180) / hist_width;

        int radius = round(hist_width * 1.4142135623730951f * (d + 1) * 0.5);  // Radius for the window
        radius = min(radius, (int)sqrt(Gauss_pyr_i.rows * Gauss_pyr_i.rows + Gauss_pyr_i.cols * Gauss_pyr_i.cols));



        int histlen = (d + 2) * (d + 2) * (n + 2);      // Length of the histogram
        // Create Histogram
        AutoBuffer<float> hist(histlen);
        memset(hist, 0, sizeof(float) * histlen);


        // ******************* Create 3 Dimensional Histogram for the Orientation, Coordinates (x,y) ******************

        for (r1 = -radius; r1 <= radius; r1++)  // Loop through each row within the chosen window
        {

            for (r2 = -radius; r2 <= radius; r2++)  // Loop through each column within the chosen window
            {


                // Get new Positions after Rotation of the window
                float c_rot = r2 * cos_t - r1 * sin_t;
                float r_rot = r2 * sin_t + r1 * cos_t;

                float rbin = r_rot + d / 2.f - 0.5f;   // Get the Width of bins for X coordinates Histogram (0 ~ 3)
                float cbin = c_rot + d / 2.f - 0.5f;  // Get the Width of bins for Y coordinates Histogram (0 ~ 3)

                // Shift the pixel (px_r, px_c)
                int px_r = kpt_r + r1;
                int px_c = kpt_c + r2;

                // Check if the pixels are within the acceptable range for columns, rows
                if (-1 < rbin && rbin < d && -1 < cbin && cbin < d &&
                    0 < px_r && px_r < Gauss_pyr_i.rows - 1 && 0 < px_c && px_c < Gauss_pyr_i.cols - 1)
                {

                    // Get Difference Between pixels in X & Y Direction to be used in the Magnitude & Orientation Computation
                    float dx = Gauss_pyr_i.at<float>(px_r, px_c + 1) - Gauss_pyr_i.at<float>(px_r, px_c - 1);
                    float dy = Gauss_pyr_i.at<float>(px_r - 1, px_c) - Gauss_pyr_i.at<float>(px_r + 1, px_c);

                    // Get New Magnitude & Direction for the obtained pixels
                    float mag = sqrt(dx * dx + dy * dy);
                    float ang = fastAtan2(dy, dx);

                    // Get the Width of Histogram Bins for the Orientation of pixels
                    float obin = (ang - kpt_ang) * (n / 360.f);     // (0 ~ 7)

                    // Get Weights for histogram bins
                    float w_G = expf(-(r_rot * r_rot + c_rot * c_rot) / (0.5f * d * d));  // - 1 / ((d / 2) ^ 2 * 2)   -> - 1 / (d ^ 2 * 0.5)
                    mag *= w_G;


                    // Initial Values for the histogram bins of Orientation & Positions (x,y)
                    int r0 = cvFloor(rbin);     // -1 0 1 2 3
                    int c0 = cvFloor(cbin);
                    int o0 = cvFloor(obin);     // (0 ~ 7)

                    rbin -= r0;
                    cbin -= c0;
                    obin -= o0;

                    //
                    if (o0 < 0)
                        o0 += n;
                    else if (o0 >= n)
                        o0 -= n;

                    // Combine them to One Histogram  0 < rbin cbin obin < 1
                    float D_rco000 = rbin * cbin * obin;
                    float D_rco001 = rbin * cbin * (1 - obin);

                    float D_rco010 = rbin * (1 - cbin) * obin;
                    float D_rco011 = rbin * (1 - cbin) * (1 - obin);

                    float D_rco100 = (1 - rbin) * cbin * obin;
                    float D_rco101 = (1 - rbin) * cbin * (1 - obin);

                    float D_rco110 = (1 - rbin) * (1 - cbin) * obin;
                    float D_rco111 = (1 - rbin) * (1 - cbin) * (1 - obin);

                    //
                    // rbin: 0 ~ 5 & r0: -1 ~ 3 , cbin: 0 ~ 5 & c0: -1 ~ 3 , obin: 0 ~ 7
                    hist[60 * (r0+1) + 10 * (c0+1) + o0] += mag * D_rco000;
                    hist[60 * (r0+1) + 10 * (c0+1) + (o0+1)] += mag * D_rco001;

                    hist[60 * (r0+1) + 10 * (c0+2) + o0] += mag * D_rco010;
                    hist[60 * (r0+1) + 10 * (c0+2) + (o0+1)] += mag * D_rco011;

                    hist[60 * (r0+2) + 10 * (c0+1) + o0] += mag * D_rco100;
                    hist[60 * (r0+2) + 10 * (c0+1) + (o0+1)] += mag * D_rco101;

                    hist[60 * (r0+2) + 10 * (c0+2) + o0] += mag * D_rco110;
                    hist[60 * (r0+2) + 10 * (c0+2) + (o0+1)] += mag * D_rco111;
                }
            }

        }


        // Obtain Histogram for the Orientation
        float Des_vec_indx[128] = { 0 };
        for (ri = 1, k = 0; ri <= 4; ri++)
            for (ci = 1; ci <= 4; ci++)
            {
                hist[60 * ri + 10 * ci + 0] += hist[60 * ri + 10 * ci + 8];
                hist[60 * ri + 10 * ci + 1] += hist[60 * ri + 10 * ci + 9];

                for (oi = 0; oi < 8; oi++)
                    Des_vec_indx[k++] = hist[60 * ri + 10 * ci + oi];
            }


        // ****************************** Select Candidate Keypoints after rotation *****************************
        float scl;
        float Des_vec_norm = 0, Des_vecThreshold;
        // Get threshold for Describtor vector
        for (k = 0; k < 128; k++)
            Des_vec_norm += Des_vec_indx[k] * Des_vec_indx[k];

        Des_vecThreshold = 0.2f * sqrtf(Des_vec_norm);

        // Get Candidate Describtors
        for (k = 0, Des_vec_norm = 0; k < 128; k++)
        {
            if (Des_vec_indx[k] > Des_vecThreshold)
                Des_vec_indx[k] = Des_vecThreshold;
            Des_vec_norm += Des_vec_indx[k] * Des_vec_indx[k];
        }

        // Normalize the Describtor vector
        scl = 1 / max(std::sqrt(Des_vec_norm), FLT_EPSILON);
        float* Des_vec_temp = Des_vec.ptr<float>(kpt_i);
        for (k = 0; k < 128; k++)
            Des_vec_temp[k] = Des_vec_indx[k] * scl;

    }
}



// ************************************************************* Draw Blobs on Images ****************************************************************

Mat Sift::Draw_Blob(Mat image, int between_lyr)
{

    // Create SIFT object
    Sift sif(between_lyr, 1.6);
    Mat output= image.clone();

    vector<keypoint> kpts;        // Keypoints vectors for both images
    Mat Desc_vec;             // Describtor vectors for both images

    sif.Detect_KeyPoints(output, kpts, Desc_vec);

    // cout << "image's detected keypoints: " << kpts.size() << endl;

    vector<KeyPoint> KPts; // KeyPoint
    keypoint::SwapToKeyPoint(kpts, KPts);     // Get Info from Keypoints detected

    // Create Blobbed Images
    Mat Detected;
    drawKeypoints(output, KPts, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return output;
    //imshow("GG", image);

}




// ************************************************************* Computational Time ****************************************************************


//void Sift::Get_time(auto start, auto stop)

//{
//    auto duration = duration_cast<microseconds>(stop - start);
//    cout << "Time taken in microseconds: ";
//    cout << duration.count() << endl;

//}



