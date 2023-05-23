#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "activecontour.h"
#include "frequency.h"
#include "histograms.h"
#include "filters.h"
#include "hough.h"
#include "sift.h"
#include "harris_operator.h"
#include "featurematching.h"
#include "thresholding.h"
#include "clustering.h"
#include "RegionGrowing.h"
#include "MeanShift.h"


#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>




QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT


public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    Mat getImage();


private slots:

    void on_actionUniform_Noise_triggered();

    void on_actionGaussian_Noise_triggered();

    void on_actionSalt_Pepper_Noise_triggered();

    void on_actionMedian_Filter_triggered();

    void on_actionGaussian_Filter_triggered();

    void on_actionAverage_Filter_triggered();

    void on_NormalizeButton_clicked();

    void on_EqualizeButton_clicked();

    void on_NoneButton_clicked();

    void on_tab1_Browse_clicked();

    void on_tab2_Browse_clicked();

    void on_GreyButton_clicked();

    void on_HistogramBox_currentTextChanged(const QString &arg1);

    void on_CumulateBox_currentTextChanged(const QString &arg1);

    //void on_spinBox_textChanged(const QString &arg1);

    void on_HistogramBox_activated(int index);

    void on_tab2_currentChanged(int index);

    void on_Median_button_clicked();

    void on_Gaussin_close_clicked();

    void on_Gaussin_button_clicked();

    void on_Average_button_clicked();

    void on_Local_button_clicked();

    void  on_actionLocal_triggered();

    void on_actionGlobal_triggered();

    void on_Global_button_clicked();

    // Edge Detection Menu
    void DisplayOutpt(Mat image);

    void on_actionPrewitt_triggered();

    void on_actionSobel_triggered();

    void on_prewitt_close_clicked();

    void on_prewitt_applay_clicked();

    void on_actionRoberts_triggered();

    void on_actionCanny_triggered();

    void on_canny_applay_clicked();

    void on_canny_close_clicked();

    void on_Sobel_clicked();

    // Noise Addition Box
    void on_noiseclose_clicked();

    void on_noiseBox1_valueChanged(double arg1);

    void on_noiseBox2_valueChanged(double arg1);

    // Tab 3
    void on_image_1_low_3_clicked();

    void on_image_1_high_3_clicked();

    void on_image_2_low_3_clicked();

    void on_image_2_high_3_clicked();

    void on_Hyperid_3_clicked();


    void on_image_1_low_r_3_valueChanged(double arg1);

    void on_image_1_high_r_3_valueChanged(double arg1);

    void on_image_2_low_r_3_valueChanged(double arg1);

    void on_image_2_high_r_3_valueChanged(double arg1);

    void on_tab3_Browse1_clicked();

    void on_tab3_Browse2_clicked();

    void on_tab4_Browse_clicked();

    void on_ObjectBox_currentTextChanged(const QString &arg1);

    void on_close_line_clicked();


    void on_close_circle_clicked();
//  void on_min_box_valueChanged(int arg1);
    void on_line_threshold_valueChanged(int arg1);
//  void on_max_box_valueChanged(int arg1);
    void on_double_minR_valueChanged(double arg1);
//  void on_double_minR_valueChanged(double arg1);
    void on_double_maxR_valueChanged(double arg1);

// Tab 5
    void on_tab5_Browse_2_clicked();
    void on_snake_2_clicked();

// Tab 7
    void on_tab7_Browse1_clicked();
    void on_tab7_Browse2_clicked();
    void Draw_Blob1();
    void Draw_Blob2();

// harris
    void on_harris_browse_clicked();

    void on_harris_button_clicked();

    void on_browse_feature_clicked();

    void on_browse_frature2_clicked();

    void on_ssd_button_clicked();

    void on_matchingbox_currentTextChanged(const QString &arg1);

    void on_feature_threshold_valueChanged(double arg1);

    void on_spinBox7_1_valueChanged(int arg1);

    void on_spinBox7_2_valueChanged(int arg1);

    void on_Threshold_browse_clicked();

    void on_thrshold_option_currentTextChanged(const QString &arg1);

    void on_global_thresholdin_button_clicked();

    void on_local_tresholding_button_clicked();

//    void on_threshold_button_clicked();

    void on_Threshold_button_clicked();

    void on_tab10_browse_clicked();

    void on_cluster_options_currentTextChanged(const QString &arg1);

    void on_KMeans_Box_valueChanged(int arg1);

    void on_done_button_clicked();

//    void on_region_threshold_textChanged(const QString &arg1);

    void on_region_threshold_valueChanged(double arg1);

    void on_meanshift_done_clicked();


    void on_spatial_val_valueChanged(double arg1);

    void on_color_val_valueChanged(double arg1);

private:
    Ui::MainWindow *ui;
    Mat colored_uploadedImage_2;
    Mat uploadedImage_1;
    Mat uploadedImage_2; 
    Mat uploadedImage_31;
    Mat uploadedImage_32;
    Mat uploadedImage_4;
    Mat uploadedImage_4_colored;
    Mat uploadedImage_7_1;
    Mat uploadedImage_7_2;
    Mat DetectedImage_7_1;
    Mat DetectedImage_7_2;
    Mat uploadedImage_10;
    Mat uploadedImage_10_Segmented;
    Mat GRAYSegmented;
    Mat NormalizedImg;
    Mat EqualizedImg;
    Mat spareimage;
    Mat upload_match_1;
    Mat upload_match_2;
    Mat match_img;

    Mat spareimage1;
    Mat uploadedImage;
    Mat originalImage;
    Mat filteredImg;
    Mat filteredImage;
    Mat output_low_image;
    Mat output_high_image;


     Mat segmented_image ;
     Mat mean_shift_image;

    Frequency freq;
    Filters kernals;
    Histograms hist;
    Hough hough;
    Sift sift;
    QString Norm_Number;
    QImage coloredimage;
    harris_operator harris;
    ActiveContour snake;
    FeatureMatching matching;
    Thresholding thresholding;
    QString thresholdingOption;
    Clustering cluster;
    RegionGrowing Region_Growing;
    MeanShift Mean_shift;


    double noise_value1=0;
    double noise_value2=0;
    int noise_menu_index;

    int averageKernal;
    int KernalSize;
    int globalT;
    int boxSize;
    int globalMax;
    int c;
    int S_lyrs = 3;
    int local;
    int global;
//  int LocalBlockSize;


    //int radius1;
    //int radius2;


};
#endif // MAINWINDOW_H
