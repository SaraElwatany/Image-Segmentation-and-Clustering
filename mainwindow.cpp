#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tab2dialog.h"


#include <QDialog>
#include <QSpinBox>
#include <QFileDialog>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>




MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{

    ui->setupUi(this);

    ui->Median_button->setVisible(0);
    ui->Gaussin->setSingleStep(2);
    ui->Gaussin->setMinimum(1);
    ui->sigma->setMinimum(1);
    ui->Average_button->setVisible(0);
    ui->Gaussin->setVisible(0);
    ui->Gaussin_button->setVisible(0);
    ui->Gaussin_close->setVisible(0);
    ui->Gaussin_3->setVisible(0);
    ui->Gaussin_4->setVisible(0);
    ui->Gaussin_2->setVisible(0);
    ui->sigma->setVisible(0);

    ui->Local_Box->setVisible(0);
    ui->Local_C->setVisible(0);
    ui->Local_Box->setMinimum(1);
    ui->Local_C->setMinimum(1);
    ui->Local_button->setVisible(0);
    ui->Global_button->setVisible(0);
    ui->Global->setVisible(0);
    ui->Global_3->setVisible(0);
    ui->Global_max->setVisible(0);
    ui->Global_4->setVisible(0);
    ui->Global_3->setMinimum(1);
    ui->Global_3->setMaximum(255);
    ui->Global_max->setMinimum(1);
    ui->Global_max->setMaximum(255);
    ui->NoiseParameters->setVisible(0);
    ui->harris_kernal->setMinimum(1);
    ui->harris_kernal->setSingleStep(2);


    // edge
    ui->prewitt_applay->setVisible(0);
    ui->prewitt_close->setVisible(0);
    ui->prewitt_label->setVisible(0);
    ui->prewitt_group->setVisible(0);
    ui->prewitt->setVisible(0);
    ui->prewitt->setMinimum(1);
    ui->prewitt->setSingleStep(2);
    ui->canny_upper->setVisible(0);
    ui->canny_lower->setVisible(0);
    ui->canny_kerna->setVisible(0);
    ui->canny_sigma->setVisible(0);
    ui->canny_group->setVisible(0);
    ui->canny_applay->setVisible(0);
    ui->canny_close->setVisible(0);
    ui->canny_label->setVisible(0);
    ui->canny_label_2->setVisible(0);
    ui->canny_label_4->setVisible(0);
    ui->canny_label_3->setVisible(0);
    ui->canny_kerna->setMinimum(1);
    ui->canny_kerna->setSingleStep(2);
    ui->canny_lower->setMinimum(1);
    ui->canny_lower->setMaximum(255);
    ui->canny_upper->setMinimum(1);
    ui->canny_upper->setMaximum(255);
    ui->canny_sigma->setMinimum(1);
    ui->iterations_2->setMaximum(2000);
    ui->radius->setMaximum(2000);
    ui->Sobel->setVisible(1);




    //Object Detection

    ui->line_label->setVisible(0);
    ui->line_threshold->setVisible(0);
    ui->line_threshold->setMaximum(255);
//  ui->min_box->setVisible(0);
    ui->min_radius->setVisible(0);
//  ui->max_box->setVisible(0);
    ui->max_radius->setVisible(0);
    ui->close_circle->setVisible(0);
    ui->close_line->setVisible(0);
    ui->double_maxR->setVisible(0);
    ui->double_minR->setVisible(0);


// threshold
  ui->threshold_value->setVisible(0);
  ui->thres->setVisible(0);
  ui->Threshold_button->setVisible(0);
  ui->block_size->setVisible(0);
  ui->blocksize->setVisible(0);
  ui->block_size->setMaximum(255);
  ui->global_thresholdin_button->setVisible(1);
  ui->global_thresholdin_button->setVisible(1);




// Clustering
    ui->KMeans_Box->setValue(1);
    ui->Agg_groupBox->setVisible(0);
    ui->KMeans_groupBox->setVisible(0);


    //Region growing

    ui->region_box->setVisible(0);
    ui->points_label->setVisible(0);
    ui->done_button->setVisible(0);
    ui->region_threshold->setVisible(0);



    //Mean Shift
    ui->meanshift->setVisible(0);
    ui->spatial_BW->setVisible(0);
    ui->color_BW->setVisible(0);
    ui->spatial_val->setVisible(0);
    ui->color_val->setVisible(0);
    ui->meanshift_done->setVisible(0);




    QPixmap pix("/home/...");
    ui->path->setPixmap(pix);


}




MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::on_tab1_Browse_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);



    uploadedImage_1 =cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
    QImage qimg(uploadedImage_1.data, uploadedImage_1.cols, uploadedImage_1.rows, uploadedImage_1.step, QImage::Format_Grayscale8);
    QPixmap output = QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(QPixmap::fromImage(qimg));
    int w= ui->tab1_image2->width();
    int h= ui->tab1_image2->height();
    ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

    if (!fileName.isEmpty())
    {

        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        ui->tab1_image1->setPixmap(pix);
        int w = ui->tab1_image1->width();
        int h = ui->tab1_image1->height();
        ui->tab1_image1->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

        uploadedImage_1 = cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        spareimage1 = uploadedImage_1;

    }
}






void MainWindow::on_actionUniform_Noise_triggered()
{
    noise_menu_index=1;
    ui->NoiseParameters->setVisible(1);
    ui->noise_param1->setText("A Value");
    ui->noise_param2->setText("B Value");

}




void MainWindow::on_actionGaussian_Noise_triggered()
{
    noise_menu_index=2;
    ui->NoiseParameters->setVisible(1);
    ui->noise_param1->setText("Mean Value");
    ui->noise_param2->setText("Sd Value");

}




void MainWindow::on_actionSalt_Pepper_Noise_triggered()
{
    noise_menu_index=3;
    ui->NoiseParameters->setVisible(1);
    ui->noise_param1->setText("PA Value");
    ui->noise_param2->setText("PB Value");

}





void MainWindow::on_noiseBox1_valueChanged(double arg1)
{
    noise_value1= ui->noiseBox1->value();

    if (noise_menu_index==1){
        Mat noisedimg1= spareimage1.clone();
        Mat img= kernals.AddUniformNoise(noisedimg1, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

    if (noise_menu_index==2){
        Mat noisedimg2= spareimage1.clone();
        Mat img= kernals.AddGaussianNoise(noisedimg2, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

    if (noise_menu_index==3){
        Mat noisedimg3= spareimage1.clone();
        Mat img= kernals.AddSaltPepperNoise(noisedimg3, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

}



void MainWindow::on_noiseBox2_valueChanged(double arg1)
{
    noise_value2= ui->noiseBox2->value();

    if (noise_menu_index==1){
        Mat noisedimg1= spareimage1.clone();
        Mat img= kernals.AddUniformNoise(noisedimg1, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

    if (noise_menu_index==2){
        Mat noisedimg2= spareimage1.clone();
        Mat img= kernals.AddGaussianNoise(noisedimg2, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

    if (noise_menu_index==3){
        Mat noisedimg3= spareimage1.clone();
        Mat img= kernals.AddSaltPepperNoise(noisedimg3, noise_value1, noise_value2);
        QImage qimg(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }
}



void MainWindow::on_noiseclose_clicked()
{
   ui->NoiseParameters->setVisible(0);
}


void MainWindow::on_actionGaussian_Filter_triggered()
{
    ui->Gaussin->setVisible(1);
    ui->Gaussin_button->setVisible(1);
    ui->Gaussin_close->setVisible(1);
    ui->sigma->setVisible(1);
    ui->Gaussin_3->setVisible(1);
    ui->Gaussin_4->setVisible(1);
    ui->Gaussin_2->setVisible(1);
    ui->Median_button->setVisible(0);
    ui->Average_button->setVisible(0);
    ui->Local_3->setVisible(0);
    ui->Local_Box->setVisible(0);
}




Mat MainWindow::getImage()
{
    return uploadedImage_1;
}





void MainWindow:: on_actionMedian_Filter_triggered()
{
      ui->Median_button->setVisible(1);
      ui->Average_button->setVisible(0);
      ui->Gaussin_button->setVisible(0);
      ui->Gaussin->setVisible(1);
      ui->Gaussin_close->setVisible(1);
      ui->Gaussin_4->setVisible(1);
      ui->Gaussin_2->setVisible(1);
      ui->Local_3->setVisible(0);
      ui->Local_Box->setVisible(0);

}






void MainWindow::on_actionAverage_Filter_triggered()
{

    ui->Average_button->setVisible(1);
    ui->Gaussin->setVisible(1);
    ui->Gaussin_close->setVisible(1);
    ui->Gaussin_2->setVisible(1);
    ui->Gaussin_4->setVisible(1);
    ui->Local_3->setVisible(0);
    ui->Local_Box->setVisible(0);
}



void MainWindow::on_actionLocal_triggered()
{
      ui->Local_Box->setVisible(1);
      ui->Local_C->setVisible(1);
      ui->Local_button->setVisible(1);
      ui->Gaussin_4->setVisible(1);
      ui->Local_2->setVisible(1);
      ui->Local_3->setVisible(1);
      ui->Gaussin_close->setVisible(1);

}





void MainWindow::on_Median_button_clicked()
{
    KernalSize= ui->Gaussin->value();
    Mat image=getImage();
    filteredImg= kernals.medianKernal(image,KernalSize);
    QImage qimg(filteredImg.data, filteredImg.cols, filteredImg.rows, filteredImg.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(output);
    int w = ui->tab1_image2->width();
    int h = ui->tab1_image2->height();
    ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
}



void MainWindow::on_Gaussin_close_clicked()
{   ui->Median_button->setVisible(0);
    ui->Average_button->setVisible(0);
    ui->Gaussin->setVisible(0);
    ui->Gaussin_button->setVisible(0);
    ui->Gaussin_close->setVisible(0);
    ui->sigma->setVisible(0);
    ui->Gaussin_3->setVisible(0);
    ui->Gaussin_4->setVisible(0);
    ui->Gaussin_2->setVisible(0);
    ui->Local_Box->setVisible(0);
    ui->Local_C->setVisible(0);
    ui->Local_button->setVisible(0);
    ui->Local_2->setVisible(0);
    ui->Local_3->setVisible(0);
    ui->Global_button->setVisible(0);
    ui->Global->setVisible(0);
    ui->Global_3->setVisible(0);
    ui->Global_max->setVisible(0);
    ui->Global_4->setVisible(0);
}




void MainWindow::on_Gaussin_button_clicked()
{
    KernalSize= ui->Gaussin->value();
    int sigma= ui->sigma->value();
    Mat image=getImage();
    filteredImg= kernals.convolve(image,KernalSize,sigma,"gaussian");
    QImage qimg(filteredImg.data, filteredImg.cols, filteredImg.rows, filteredImg.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(output);
    int w = ui->tab1_image2->width();
    int h = ui->tab1_image2->height();
    ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

}




void MainWindow::on_Average_button_clicked()
{
        averageKernal= ui->Gaussin->value();
        Mat image=getImage();
        filteredImg= kernals.convolve(image,averageKernal,1,"average");
        QImage qimg(filteredImg.data, filteredImg.cols, filteredImg.rows, filteredImg.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->tab1_image2->setPixmap(output);
        int w = ui->tab1_image2->width();
        int h = ui->tab1_image2->height();
        ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
}





void MainWindow::on_Local_button_clicked()
{
    boxSize=ui->Local_Box->value();
    c=ui->Local_C->value();
    Mat image=getImage();
    filteredImg= kernals.localThresholding(image,boxSize,c);
    QImage qimg( filteredImg.data, filteredImg.cols, filteredImg.rows, filteredImg.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(output);
    int w = ui->tab1_image2->width();
    int h = ui->tab1_image2->height();
    ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

}




void MainWindow:: on_actionGlobal_triggered(){
    ui->Global->setVisible(1);
    ui->Global_3->setVisible(1);
    ui->Global_max->setVisible(1);
    ui->Global_4->setVisible(1);
    ui->Global_button->setVisible(1);
    ui->Gaussin_4->setVisible(1);
    ui->Gaussin_close->setVisible(1);


}





void MainWindow::on_Global_button_clicked()
{
    globalT=ui->Global_3->value();
    globalMax=ui->Global_max->value();
    filteredImg=kernals.globalThresholding(uploadedImage_1,globalT,globalMax);
    DisplayOutpt(filteredImg);
}






// Edge Detection
void MainWindow::DisplayOutpt(Mat image){
    QImage qimg( image.data, image.cols, image.rows, image.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(output);
    int w = ui->tab1_image2->width();
    int h = ui->tab1_image2->height();
    ui->tab1_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
}







void MainWindow::on_actionPrewitt_triggered(){
    ui->prewitt_applay->setVisible(1);
    ui->prewitt_close->setVisible(1);
    ui->prewitt_label->setVisible(1);
    ui->prewitt_group->setVisible(1);
    ui->prewitt->setVisible(1);
    ui->Sobel->setVisible(0);

}






void MainWindow::on_prewitt_close_clicked()
{
    ui->prewitt_applay->setVisible(0);
    ui->prewitt_close->setVisible(0);
    ui->prewitt_label->setVisible(0);
    ui->prewitt_group->setVisible(0);
    ui->prewitt->setVisible(0);
    ui->Sobel->setVisible(0);
}







void MainWindow::on_prewitt_applay_clicked()

{   KernalSize=ui->prewitt->value();
    filteredImg=kernals.prewittEdgeDetector(uploadedImage_1,KernalSize);
    DisplayOutpt(filteredImg);
}






void MainWindow:: on_actionRoberts_triggered(){

    filteredImg=kernals.robert(uploadedImage_1);
    DisplayOutpt(filteredImg);
}







void MainWindow:: on_actionCanny_triggered(){
     ui->canny_upper->setVisible(1);
     ui->canny_lower->setVisible(1);
     ui->canny_kerna->setVisible(1);
     ui->canny_sigma->setVisible(1);
     ui->canny_group->setVisible(1);
     ui->canny_applay->setVisible(1);
     ui->canny_close->setVisible(1);
     ui->canny_label->setVisible(1);
     ui->canny_label_2->setVisible(1);
     ui->canny_label_4->setVisible(1);
     ui->canny_label_3->setVisible(1);

}





void MainWindow::on_actionSobel_triggered(){
    ui->prewitt_close->setVisible(1);
    ui->prewitt_label->setVisible(1);
    ui->prewitt_group->setVisible(1);
    ui->prewitt->setVisible(1);
    ui->Sobel->setVisible(1);
    ui->prewitt_applay->setVisible(0);
}







void MainWindow::on_canny_applay_clicked()
{
    int upper_thresh=ui->canny_upper->value();
    int lower_thresh=ui->canny_lower->value();
    KernalSize=ui->canny_kerna->value();
    int sigma=ui->canny_sigma->value();
    filteredImg=kernals.cannyEdgeDetector(uploadedImage_1, lower_thresh, upper_thresh, KernalSize, sigma);
    DisplayOutpt(filteredImg);
}








void MainWindow::on_canny_close_clicked()
{
    ui->canny_upper->setVisible(0);
    ui->canny_lower->setVisible(0);
    ui->canny_kerna->setVisible(0);
    ui->canny_sigma->setVisible(0);
    ui->canny_group->setVisible(0);
    ui->canny_applay->setVisible(0);
    ui->canny_close->setVisible(0);
    ui->canny_label->setVisible(0);
    ui->canny_label_2->setVisible(0);
    ui->canny_label_4->setVisible(0);
    ui->canny_label_3->setVisible(0);

}







void MainWindow::on_Sobel_clicked()
{  KernalSize=ui->prewitt->value();
    filteredImg=kernals.sobelFilter(uploadedImage_1,KernalSize);
    DisplayOutpt(filteredImg);
}









//*******************************************************************Second Tab***************************************************************************

void MainWindow::on_tab2_Browse_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"), 0, QFileDialog::DontUseNativeDialog);

    if (!fileName.isEmpty())
    {
        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        coloredimage = image;
        ui->tab2_image1->setPixmap(QPixmap::fromImage(image));
        int w = ui->tab2_image1->width();
        int h = ui->tab2_image1->height();
        ui->tab2_image1->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

        uploadedImage_2= imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        colored_uploadedImage_2= imread(fileName.toStdString(),cv::IMREAD_ANYCOLOR);
    }

}






void MainWindow::on_EqualizeButton_clicked()
{
    EqualizedImg = freq.Equalize(uploadedImage_2);
    spareimage = EqualizedImg;
    QImage qimg(EqualizedImg.data, EqualizedImg.cols, EqualizedImg.rows, EqualizedImg.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab2_image2->setPixmap(output);
    int w = ui->tab2_image2->width();
    int h = ui->tab2_image2->height();
    ui->tab2_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    on_HistogramBox_currentTextChanged("");

}







void MainWindow::on_NormalizeButton_clicked()
{

    //Norm_Number= ui->spinBox->text();
    NormalizedImg = freq.Normalize(uploadedImage_2);
    spareimage = NormalizedImg;
    QImage qimg(NormalizedImg.data, NormalizedImg.cols, NormalizedImg.rows, NormalizedImg.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab2_image2->setPixmap(output);
    int w = ui->tab2_image2->width();
    int h = ui->tab2_image2->height();
    ui->tab2_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    on_HistogramBox_currentTextChanged("");
}







void MainWindow::on_NoneButton_clicked()
{
    spareimage = uploadedImage_2;
    QPixmap image=QPixmap::fromImage(coloredimage);
    ui->tab2_image2->setPixmap(image);
    int w = ui->tab2_image2->width();
    int h = ui->tab2_image2->height();
    ui->tab2_image2->setPixmap(image.scaled(w,h,Qt::KeepAspectRatio));
    on_HistogramBox_currentTextChanged("");

}







void MainWindow::on_GreyButton_clicked()
{
    spareimage = uploadedImage_2;
    QImage qimg(uploadedImage_2.data, uploadedImage_2.cols, uploadedImage_2.rows, uploadedImage_2.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab2_image2->setPixmap(output);
    int w = ui->tab2_image2->width();
    int h = ui->tab2_image2->height();
    ui->tab2_image2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    on_HistogramBox_currentTextChanged("");
}







void MainWindow::on_HistogramBox_currentTextChanged(const QString &arg1)
{
    QString get_text = ui->HistogramBox->currentText();


    if (get_text == "Grey-Channel"){

        cv::Mat histogram = freq.GetHistogram(spareimage);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }


    if (get_text == "R-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 2);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }


    if (get_text == "G-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 1);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

    if (get_text == "B-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 0);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

}






void MainWindow::on_CumulateBox_currentTextChanged(const QString &arg1)
{
    QString get_text = ui->CumulateBox->currentText();

    if (get_text == "Grey-Channel"){

        cv::Mat histogram = freq.GetHistogram(spareimage);
        histogram= freq.CumulativeHistogram(histogram);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
        //ui->CumulateBox->setCurrentIndex(0);
    }


    if (get_text == "R-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 2);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.CumulativeHistogram(histogram);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }


    if (get_text == "G-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 1);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.CumulativeHistogram(histogram);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }


    if (get_text == "B-Channel"){

        Mat channel = freq.ImageSplit(colored_uploadedImage_2, 0);
        Mat histogram = freq.GetHistogram(channel);
        histogram= freq.CumulativeHistogram(histogram);
        histogram= freq.DrawHistogram(histogram);
        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
        QPixmap output= QPixmap::fromImage(qimg);
        ui->tab2_image3->setPixmap(output);
        int w = ui->tab2_image3->width();
        int h = ui->tab2_image3->height();
        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }

}




/*void MainWindow::on_spinBox_textChanged(const QString &arg1)
{
    on_NormalizeButton_clicked();
}*/




void MainWindow::on_HistogramBox_activated(int index)
{
    on_HistogramBox_currentTextChanged("");
}



void MainWindow::on_tab2_currentChanged(int index)
{
    int tab = ui->tab2->currentIndex();

    if (tab==0){
        ui->menubar->show();
    }

    else{
        ui->menubar->hide();
    }

}





//**********************************************************************Third Tab************************************************************************



void MainWindow::on_image_1_low_3_clicked()
{
    int radius=ui->image_1_low_r_3->value();
    Mat DFT_image;
    hist.DFT(uploadedImage_31, DFT_image);
    Mat filter_high=DFT_image.clone();
    hist.lowpassFilter(filter_high, radius);
    hist.crop_and_rearrange(DFT_image);
    mulSpectrums(DFT_image, filter_high, DFT_image, 0);
    hist.crop_and_rearrange(DFT_image);
//  Mat output_image;
    hist.invers_DFT(DFT_image,output_low_image);
    QImage qimg(output_low_image.data, output_low_image.cols, output_low_image.rows, output_low_image.step, QImage::Format_Grayscale8);
    QPixmap output= QPixmap::fromImage(qimg);
    ui->tab3_image1_3->setPixmap(output);
    int w = ui->tab3_image1_3->width();
    int h = ui->tab3_image1_3->height();
    ui->tab3_image1_3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    /*namedWindow("gray3", WINDOW_AUTOSIZE);
    imshow("gray3", output_low_image);*/
}




void MainWindow::on_image_1_high_3_clicked()
{
    int radius=ui->image_1_high_r_3->value();
    Mat DFT_image;
    hist.DFT(uploadedImage_31, DFT_image);
    Mat filter_high=DFT_image.clone();
    hist.highpassFilter(filter_high, radius);
    hist.crop_and_rearrange(DFT_image);
    mulSpectrums(DFT_image, filter_high, DFT_image, 0);
    hist.crop_and_rearrange(DFT_image);
//  Mat output_image;
    hist.invers_DFT(DFT_image,output_high_image);
    QImage qimg(output_high_image.data, output_high_image.cols, output_high_image.rows, output_high_image.step, QImage::Format_Grayscale8);
    QPixmap output= QPixmap::fromImage(qimg);
    ui->tab3_image1_3->setPixmap(output);
    int w = ui->tab3_image1_3->width();
    int h = ui->tab3_image1_3->height();
    ui->tab3_image1_3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    /*namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", output_high_image);*/
}




void MainWindow::on_image_2_low_3_clicked()
{
    int radius=ui->image_2_low_r_3->value();
    Mat DFT_image;
    hist.DFT(uploadedImage_32, DFT_image);
    Mat filter_high=DFT_image.clone();
    hist.lowpassFilter(filter_high, radius);
    hist.crop_and_rearrange(DFT_image);
    mulSpectrums(DFT_image, filter_high, DFT_image, 0);
    hist.crop_and_rearrange(DFT_image);
//  Mat output_image;
    hist.invers_DFT(DFT_image,output_low_image);
    QImage qimg(output_low_image.data, output_low_image.cols, output_low_image.rows, output_low_image.step, QImage::Format_Grayscale8);
    QPixmap output= QPixmap::fromImage(qimg);
    ui->tab3_image2_3->setPixmap(output);
    int w = ui->tab3_image2_3->width();
    int h = ui->tab3_image2_3->height();
    ui->tab3_image2_3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    /*namedWindow("gray2", WINDOW_AUTOSIZE);
    imshow("gray2", output_low_image);*/
}





void MainWindow::on_image_2_high_3_clicked()
{
    int radius=ui->image_2_high_r_3->value();
    Mat DFT_image;
    hist.DFT(uploadedImage_32, DFT_image);
    Mat filter_high=DFT_image.clone();
    hist.highpassFilter(filter_high, radius);
    hist.crop_and_rearrange(DFT_image);
    mulSpectrums(DFT_image, filter_high, DFT_image, 0);
    hist.crop_and_rearrange(DFT_image);
//  Mat output_image;
    hist.invers_DFT(DFT_image,output_high_image);
    QImage qimg(output_high_image.data, output_high_image.cols, output_high_image.rows, output_high_image.step, QImage::Format_Grayscale8);
    QPixmap output= QPixmap::fromImage(qimg);
    ui->tab3_image2_3->setPixmap(output);
    int w = ui->tab3_image2_3->width();
    int h = ui->tab3_image2_3->height();
    ui->tab3_image2_3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    /*namedWindow("gray1", WINDOW_AUTOSIZE);
    imshow("gray1", output_high_image);*/
}




void MainWindow::on_Hyperid_3_clicked()
{
    Mat output=hist.Hyprid_images(output_low_image,output_high_image);
    QImage qimg(output.data, output.cols, output.rows, output.step, QImage::Format_Grayscale8);
    QPixmap output2= QPixmap::fromImage(qimg);
    ui->tab3_image3_3->setPixmap(output2);
    int w = ui->tab3_image3_3->width();
    int h = ui->tab3_image3_3->height();
    ui->tab3_image3_3->setPixmap(output2.scaled(w,h,Qt::KeepAspectRatio));
    /*namedWindow("hybrid", WINDOW_AUTOSIZE);
    imshow("hybrid", output);*/
}





void MainWindow::on_image_1_low_r_3_valueChanged(double arg1)
{
    on_image_1_low_3_clicked();
}





void MainWindow::on_image_1_high_r_3_valueChanged(double arg1)
{
    on_image_1_high_3_clicked();
}





void MainWindow::on_image_2_low_r_3_valueChanged(double arg1)
{
    on_image_2_low_3_clicked();
}




void MainWindow::on_image_2_high_r_3_valueChanged(double arg1)
{
    on_image_2_high_3_clicked();
}




void MainWindow::on_tab3_Browse1_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);


    if (!fileName.isEmpty())
    {
        uploadedImage_31 = cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        QImage qimg(uploadedImage_31.data, uploadedImage_31.cols, uploadedImage_31.rows, uploadedImage_31.step, QImage::Format_Grayscale8);

        QPixmap pix = QPixmap::fromImage(qimg);
        ui->tab3_image1_3->setPixmap(pix);
        int w = ui->tab3_image1_3->width();
        int h = ui->tab3_image1_3->height();
        ui->tab3_image1_3->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

    }
}







void MainWindow::on_tab3_Browse2_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);


    if (!fileName.isEmpty())
    {


        uploadedImage_32 = cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        QImage qimg(uploadedImage_32.data, uploadedImage_32.cols, uploadedImage_32.rows, uploadedImage_32.step, QImage::Format_Grayscale8);

        QPixmap pix = QPixmap::fromImage(qimg);
        ui->tab3_image2_3->setPixmap(pix);
        int w = ui->tab3_image2_3->width();
        int h = ui->tab3_image2_3->height();
        ui->tab3_image2_3->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

    }
}








//**********************************************************************Fourth Tab************************************************************************
void MainWindow::on_tab4_Browse_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);


    if (!fileName.isEmpty())
    {


        uploadedImage_4 = cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        uploadedImage_4_colored = cv::imread(fileName.toStdString(),cv::IMREAD_ANYCOLOR);
        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        ui->tab4_image1->setPixmap(pix);
        int w = ui->tab4_image1->width();
        int h = ui->tab4_image1->height();
        ui->tab4_image1->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

    }







//        cv::Mat histogram = freq.GetHistogram(spareimage);
//        histogram= freq.CumulativeHistogram(histogram);
//        histogram= freq.DrawHistogram(histogram);
//        QImage qimg(histogram.data, histogram.cols, histogram.rows, histogram.step, QImage::Format_Grayscale8);
//        QPixmap output= QPixmap::fromImage(qimg);
//        ui->tab2_image3->setPixmap(output);
//        int w = ui->tab2_image3->width();
//        int h = ui->tab2_image3->height();
//        ui->tab2_image3->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
//        ui->CumulateBox->setCurrentIndex(0);


}

void MainWindow::on_ObjectBox_currentTextChanged(const QString &arg1)
{

    QString option = ui->ObjectBox->currentText();

    if (option == "Line Detection"){
        ui->line_label->setVisible(1);
        ui->line_threshold->setVisible(1);
        ui->close_line->setVisible(1);


        Mat image_edge;
        Mat blurred;

        blur(uploadedImage_4,blurred,Size(5,5));
        Canny(blurred,image_edge,100,150,3);
        int w=image_edge.cols;
        int h=image_edge.rows;

        hough.Line_Transform(image_edge.data,w,h);
        int threshold=ui->line_threshold->value();

        vector<pair<pair<int,int>,pair<int,int>>> lines=hough.GetLines(threshold);

        Mat output_image;
        vector<pair<pair<int, int>,pair<int, int> > >::iterator it;

        Mat detected = hough.draw_lines(uploadedImage_4,lines);
        QImage::Format format=QImage::Format_Grayscale8;
        int bpp=detected.channels();
        if(bpp==3)
            format=QImage::Format_RGB888;
        QImage img(detected.cols,detected.rows,format);
        uchar *sptr,*dptr;
        int linesize=detected.cols*bpp;
        for(int y=0;y<detected.rows;y++){
            sptr=detected.ptr(y);
            dptr=img.scanLine(y);
            memcpy(dptr,sptr,linesize);
        }
        if(bpp==3)
           QPixmap::fromImage(img.rgbSwapped());
        QPixmap pix = QPixmap::fromImage(img);
        ui->tab4_image2->setPixmap(pix);
        int width = ui->tab4_image2->width();
        int height = ui->tab4_image2->height();
        ui->tab4_image2->setPixmap(pix.scaled(width,height,Qt::KeepAspectRatio));
        
    }

    

    if (option == "Circle Detection"){
        ui->double_maxR->setVisible(1);
        ui->double_minR->setVisible(1);

        ui->line_label->setVisible(0);
        ui->line_threshold->setVisible(0);
        ui->min_radius->setVisible(1);
        ui->max_radius->setVisible(1);
        ui->close_circle->setVisible(1);



        Mat image_edge;
        Mat blurred;

        blur(uploadedImage_4,blurred,Size(5,5));
        Canny(blurred,image_edge,100,150,3);

         vector<Circle> Circles;
         double min_radius = ui->double_minR->value();
         double max_radius = ui->double_maxR->value();
         hough.Circle_Transform(image_edge, Circles, min_radius, max_radius);
         Mat resultImgHough = hough.drawCircles(uploadedImage_4, Circles, (int) Circles.size());
         Mat detected = resultImgHough;
         QImage::Format format=QImage::Format_Grayscale8;
         int bpp=detected.channels();
         if(bpp==3)
            format=QImage::Format_RGB888;
         QImage img(detected.cols,detected.rows,format);
         uchar *sptr,*dptr;
         int linesize=detected.cols*bpp;
         for(int y=0;y<detected.rows;y++){
             sptr=detected.ptr(y);
             dptr=img.scanLine(y);
             memcpy(dptr,sptr,linesize);
         }
         if(bpp==3)
            QPixmap::fromImage(img.rgbSwapped());
         QPixmap pix = QPixmap::fromImage(img);
         ui->tab4_image2->setPixmap(pix);
         int width = ui->tab4_image2->width();
         int height = ui->tab4_image2->height();
         ui->tab4_image2->setPixmap(pix.scaled(width,height,Qt::KeepAspectRatio));
    }


    if (option == "Ellipse Detection"){

        Mat detected = hough.Preprocessing(uploadedImage_4_colored, uploadedImage_4);
        QImage::Format format=QImage::Format_Grayscale8;
        int bpp=detected.channels();
        if(bpp==3)
            format=QImage::Format_RGB888;
        QImage img(detected.cols,detected.rows,format);
        uchar *sptr,*dptr;
        int linesize=detected.cols*bpp;
        for(int y=0;y<detected.rows;y++){
            sptr=detected.ptr(y);
            dptr=img.scanLine(y);
            memcpy(dptr,sptr,linesize);
        }
        if(bpp==3)
            QPixmap::fromImage(img.rgbSwapped());
        QPixmap pix = QPixmap::fromImage(img);
        ui->tab4_image2->setPixmap(pix);
        int w = ui->tab4_image2->width();
        int h = ui->tab4_image2->height();
        ui->tab4_image2->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

    }


}

void MainWindow::on_close_line_clicked()
{
    ui->line_label->setVisible(0);
    ui->line_threshold->setVisible(0);
    ui->close_line->setVisible(0);


}



void MainWindow::on_close_circle_clicked()
{

//  ui->min_box->setVisible(0);
//  ui->max_box->setVisible(0);
    ui->min_radius->setVisible(0);
    ui->max_radius->setVisible(0);
    ui->close_circle->setVisible(0);
    ui->double_maxR->setVisible(0);
    ui->double_minR->setVisible(0);

}
//void MainWindow::on_min_box_valueChanged(int arg1)
//{
//    on_ObjectBox_currentTextChanged("test");
//}
//void MainWindow::on_max_box_valueChanged(int arg1)
//{
//    on_ObjectBox_currentTextChanged("test");
//}

void MainWindow::on_line_threshold_valueChanged(int arg1)
{
    on_ObjectBox_currentTextChanged("test");
}






void MainWindow::on_double_minR_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");


}


void MainWindow::on_double_maxR_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");

}


//**********************************************************************Fifth Tab************************************************************************


void MainWindow::on_tab5_Browse_2_clicked()
{
    QFileDialog dialog(this);
        dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
        dialog.setViewMode(QFileDialog::Detail);
        QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
            tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);



        uploadedImage_1 =cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        QImage qimg(uploadedImage_1.data, uploadedImage_1.cols, uploadedImage_1.rows, uploadedImage_1.step, QImage::Format_Grayscale8);
        QPixmap output = QPixmap::fromImage(qimg);
        ui->tab5_image2_2->setPixmap(QPixmap::fromImage(qimg));
        int w= ui->tab5_image2_2->width();
        int h= ui->tab5_image2_2->height();
        ui->tab5_image2_2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

        if (!fileName.isEmpty())
        {

            QImage image(fileName);
            QPixmap pix = QPixmap::fromImage(image);
            ui->tab5_image1_2->setPixmap(pix);
            int w = ui->tab5_image1_2->width();
            int h = ui->tab5_image1_2->height();
            ui->tab5_image1_2->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

            uploadedImage_1 = cv::imread(fileName.toStdString(),COLOR_BGR2BGRA);
            spareimage1 = uploadedImage_1;

        }
}




void MainWindow::on_snake_2_clicked()
{
    double beta= ui->beta_2->value();
    double alpha = ui->alpha_2->value();
    double gamma= ui->gamma_2->value();
    int radius= ui->radius->value();
    int numberOfIterations= ui->iterations_2->value();
    Mat result;
    Point center(uploadedImage_1.cols / 2, uploadedImage_1.rows / 2);
    std::vector<Point> curve=snake.contourUpdating(uploadedImage_1,result,center,radius,numberOfIterations, alpha, beta, gamma);
    // from 1418 to 1423 -> print lines to show the output frome chain code.
    std::vector<int> chainCode;
    chainCode=snake.calculateChainCode(curve);
    std::cout << "Chain code: ";
        for (int i = 0; i < chainCode.size(); i++) {
            std::cout << chainCode[i] << " "<< std::endl;
        }
    QImage qimg(result.data, result.cols, result.rows, result.step, QImage::Format_Grayscale8);
    QPixmap output=QPixmap::fromImage(qimg);
    ui->tab1_image2->setPixmap(output);
    int w = ui->tab1_image2->width();
    int h = ui->tab1_image2->height();
    ui->tab5_image2_2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
//  cv::imshow("output",result);
//  waitKey();

}




//******************************************************************** SIFT Tab ************************************************************************
void MainWindow::on_tab7_Browse1_clicked()
{

    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);

    if (!fileName.isEmpty())
    {


        uploadedImage_7_1 = cv::imread(fileName.toStdString(),cv::IMREAD_ANYCOLOR);
        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        ui->tab7_image11->setPixmap(pix);
        int w = ui->tab7_image11->width();
        int h = ui->tab7_image11->height();
        ui->tab7_image11->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));



        // Draw detected Features
        auto start = high_resolution_clock::now();
        DetectedImage_7_1 = sift.Draw_Blob(uploadedImage_7_1, S_lyrs);
        // Get Computational Time
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken in microseconds for first image: ";
        cout << duration.count() << endl;

        cvtColor(DetectedImage_7_1, DetectedImage_7_1, cv::COLOR_BGR2RGB);
        QImage img2 = QImage((uchar*)DetectedImage_7_1.data, DetectedImage_7_1.cols, DetectedImage_7_1.rows, DetectedImage_7_1.step, QImage::Format_RGB888);


        QPixmap pix2 = QPixmap::fromImage(img2);
        ui->tab7_image12->setPixmap(pix2);
        int width = ui->tab7_image12->width();
        int height = ui->tab7_image12->height();
        ui->tab7_image12->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));

    }



}





void MainWindow::on_tab7_Browse2_clicked()
{

    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);

    if (!fileName.isEmpty())
    {


        uploadedImage_7_2 = cv::imread(fileName.toStdString(),cv::IMREAD_ANYCOLOR);
        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        ui->tab7_image21->setPixmap(pix);
        int w = ui->tab7_image21->width();
        int h = ui->tab7_image21->height();
        ui->tab7_image21->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));


        // Draw detected Features
        auto start = high_resolution_clock::now();
        DetectedImage_7_2 = sift.Draw_Blob(uploadedImage_7_2, S_lyrs);

        // Get Computational Time
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken in microseconds for second image: ";
        cout << duration.count() << endl;

        cvtColor(DetectedImage_7_2, DetectedImage_7_2, cv::COLOR_BGR2RGB);
        QImage img2 = QImage((uchar*)DetectedImage_7_2.data, DetectedImage_7_2.cols, DetectedImage_7_2.rows, DetectedImage_7_2.step, QImage::Format_RGB888);



        QPixmap pix2 = QPixmap::fromImage(img2);
        ui->tab7_image22->setPixmap(pix2);
        int width = ui->tab7_image22->width();
        int height = ui->tab7_image22->height();
        ui->tab7_image22->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));

    }


}



void MainWindow::Draw_Blob1(){

    // Draw detected Features
    auto start = high_resolution_clock::now();
    DetectedImage_7_1 = sift.Draw_Blob(uploadedImage_7_1, S_lyrs);
    // Get Computational Time
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken in microseconds for first image: ";
    cout << duration.count() << endl;

    cvtColor(DetectedImage_7_1, DetectedImage_7_1, cv::COLOR_BGR2RGB);
    QImage img2 = QImage((uchar*)DetectedImage_7_1.data, DetectedImage_7_1.cols, DetectedImage_7_1.rows, DetectedImage_7_1.step, QImage::Format_RGB888);

    QPixmap pix2 = QPixmap::fromImage(img2);
    ui->tab7_image12->setPixmap(pix2);
    int width = ui->tab7_image12->width();
    int height = ui->tab7_image12->height();
    ui->tab7_image12->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));
}




void MainWindow::Draw_Blob2(){

    // Draw detected Features
    auto start = high_resolution_clock::now();
    DetectedImage_7_2 = sift.Draw_Blob(uploadedImage_7_2, S_lyrs);

    // Get Computational Time
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken in microseconds for second image: ";
    cout << duration.count() << endl;

    cvtColor(DetectedImage_7_2, DetectedImage_7_2, cv::COLOR_BGR2RGB);
    QImage img2 = QImage((uchar*)DetectedImage_7_2.data, DetectedImage_7_2.cols, DetectedImage_7_2.rows, DetectedImage_7_2.step, QImage::Format_RGB888);


    QPixmap pix2 = QPixmap::fromImage(img2);
    ui->tab7_image22->setPixmap(pix2);
    int width = ui->tab7_image22->width();
    int height = ui->tab7_image22->height();
    ui->tab7_image22->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));

}


void MainWindow::on_spinBox7_1_valueChanged(int arg1)
{
    S_lyrs = arg1;
    Draw_Blob1();
}




void MainWindow::on_spinBox7_2_valueChanged(int arg1)
{
    S_lyrs = arg1;
    Draw_Blob2();
}





//**********************************************************************Harris Tab************************************************************************

void MainWindow::on_harris_browse_clicked()
{
    QFileDialog dialog(this);
        dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
        dialog.setViewMode(QFileDialog::Detail);
        QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
            tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);

        uploadedImage_1 =cv::imread(fileName.toStdString());
        QImage qimg(uploadedImage_1.data, uploadedImage_1.cols, uploadedImage_1.rows, uploadedImage_1.step, QImage::Format_RGB888);
        QPixmap output = QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(QPixmap::fromImage(qimg));
        int w= ui->harris_output->width();
        int h= ui->harris_output->height();
        ui->harris_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
        if (!fileName.isEmpty())
        {

            QImage image(fileName);
            QPixmap pix = QPixmap::fromImage(image);
            ui->harris_input->setPixmap(pix);
            int w = ui->harris_input->width();
            int h = ui->harris_input->height();
            ui->harris_input->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

            uploadedImage_1 = cv::imread(fileName.toStdString());
            spareimage1 = uploadedImage_1;

        }
}





void MainWindow::on_harris_button_clicked()
{

            int kernal_size= ui->harris_kernal->value();
             double thresholding = ui->thresholding->value();
            double alpha=0.05;
            clock_t start, end;
            start = clock();
            Mat gray=harris.convertToGray(uploadedImage_1);
            Mat result= harris.harrisCorner(gray,kernal_size,alpha);
            harris.setThreshold(result,thresholding);
            Mat localMax;
            harris.getLocalMax(result, localMax);
            Mat harris_output= harris.drawPointsToImg(uploadedImage_1, localMax);
            end = clock();
            // Calculating total time taken by the program.
            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            QString time_taken_str = QString("Time taken by program is: %1 sec").arg(QString::number(time_taken, 'f', 5));
                ui->Time->setText(time_taken_str);
//            cout << "Time taken by program is : " << fixed
//                 << time_taken << setprecision(5);
//            cout << " sec " << endl;

            QImage qimg(harris_output.data, harris_output.cols, harris_output.rows, harris_output.step, QImage::Format_RGB888);
            QPixmap output=QPixmap::fromImage(qimg);
            ui->harris_output->setPixmap(output);
            int w = ui->harris_output->width();
            int h = ui->harris_output->height();
            ui->harris_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

}


//**************************************************************** Feature_matching Tab ************************************************************************



void MainWindow::on_browse_feature_clicked()
{
    {
        QFileDialog dialog(this);
            dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
            dialog.setViewMode(QFileDialog::Detail);
            QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
                tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);



            upload_match_1 =cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
            QImage qimg(upload_match_1.data, upload_match_1.cols, upload_match_1.rows, upload_match_1.step, QImage::Format_Grayscale8);
            QPixmap output = QPixmap::fromImage(qimg);
            ui->feature_1->setPixmap(QPixmap::fromImage(qimg));
            int w= ui->feature_1->width();
            int h= ui->feature_1->height();
            ui->feature_1->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

            if (!fileName.isEmpty())
            {

                QImage image(fileName);
                QPixmap pix = QPixmap::fromImage(image);
                ui->feature_1->setPixmap(pix);
                int w = ui->feature_1->width();
                int h = ui->feature_1->height();
                ui->feature_1->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

//                upload_match = cv::imread(fileName.toStdString(),COLOR_BGR2BGRA);
//                spareimage1 = upload_match;

            }
    }
}




void MainWindow::on_browse_frature2_clicked()
{
    {
        QFileDialog dialog(this);
            dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
            dialog.setViewMode(QFileDialog::Detail);
            QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
                tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);



            upload_match_2 =cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
            QImage qimg(upload_match_2.data, upload_match_2.cols, upload_match_2.rows, upload_match_2.step, QImage::Format_Grayscale8);
            QPixmap output = QPixmap::fromImage(qimg);
            ui->feature_2->setPixmap(QPixmap::fromImage(qimg));
            int w= ui->feature_2->width();
            int h= ui->feature_2->height();
            ui->feature_2->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

            if (!fileName.isEmpty())
            {

                QImage image(fileName);
                QPixmap pix = QPixmap::fromImage(image);
                ui->feature_2->setPixmap(pix);
                int w = ui->feature_2->width();
                int h = ui->feature_2->height();
                ui->feature_2->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

            }
    }
}





void MainWindow::on_ssd_button_clicked()
{
    ui->feature_threshold->setVisible(0);
    ui->ssd_threshold->setVisible(0);
    ui->ssd_button->setVisible(0);


}


void MainWindow::on_matchingbox_currentTextChanged(const QString &arg1)
{
    QString option_matching = ui->matchingbox->currentText();
    if (option_matching == "Sum of squard differenced"){
        ui->feature_threshold->setVisible(1);
        ui->ssd_threshold->setVisible(1);
        ui->ssd_button->setVisible(1);

            Ptr<SIFT> sift = SIFT::create();
            vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1, descriptors2;
            sift->detectAndCompute(upload_match_1, noArray(), keypoints1, descriptors1);
            sift->detectAndCompute(upload_match_2, noArray(), keypoints2, descriptors2);
            double threshold = ui->feature_threshold->value();
            vector<DMatch> matches=matching.match_features(descriptors1,descriptors2,threshold);
            drawMatches(upload_match_1, keypoints1, upload_match_2, keypoints2, matches, match_img, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            QImage::Format format=QImage::Format_Grayscale8;
            int bpp=match_img.channels();
            if(bpp==3)
                format=QImage::Format_RGB888;
            QImage img(match_img.cols,match_img.rows,format);
            uchar *sptr,*dptr;
            int linesize=match_img.cols*bpp;
            for(int y=0;y<match_img.rows;y++){
                sptr=match_img.ptr(y);
                dptr=img.scanLine(y);
                memcpy(dptr,sptr,linesize);
            }
            if(bpp==3)
               QPixmap::fromImage(img.rgbSwapped());
            QPixmap pix = QPixmap::fromImage(img);
            ui->ssd_output->setPixmap(pix);
            int width = ui->ssd_output->width();
            int height = ui->ssd_output->height();
            ui->ssd_output->setPixmap(pix.scaled(width,height,Qt::KeepAspectRatio));


    }
     if (option_matching == "Nomlized Correlation"){
         Ptr<SIFT> sift = SIFT::create();
         vector<KeyPoint> keypoints1, keypoints2;
         Mat descriptors1, descriptors2;
         cv::resize(upload_match_1, upload_match_1, Size(256, 256));
         cv::resize(upload_match_2, upload_match_2, Size(256, 256));
         sift->detectAndCompute(upload_match_1, noArray(), keypoints1, descriptors1);
         sift->detectAndCompute(upload_match_2, noArray(), keypoints2, descriptors2);
         //Calls the feature_matching_temp() function with method="NCC"
         vector<DMatch> matched_features = matching.feature_matching_temp(descriptors1, descriptors2,"ncc");

         //sort the features in order to identify the best matches
         sort(matched_features.begin(), matched_features.end(), [](DMatch match1, DMatch match2) {
         return match1.distance > match2.distance; });

         //visualizes the results
         Mat result ;

         vector<DMatch> best_matches;

         for(int i =0 ; i<30; i++){
             best_matches.push_back(matched_features[i]);
         }

         drawMatches(upload_match_1, keypoints1, upload_match_2, keypoints1,best_matches, result,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
         QImage qimg(result.data, result.cols, result.rows, result.step, QImage::Format_RGB888);
         QPixmap output=QPixmap::fromImage(qimg);
         ui->ssd_output->setPixmap(output);
         int w = ui->ssd_output->width();
         int h = ui->ssd_output->height();
         ui->ssd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
     }


}


void MainWindow::on_feature_threshold_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");


}




//**********************************************************************Thrsholding Methods Tab************************************************************************


void MainWindow::on_Threshold_browse_clicked()
{
    QFileDialog dialog(this);
        dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
        dialog.setViewMode(QFileDialog::Detail);
        QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
            tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);

        uploadedImage_1 =cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
        QImage qimg(uploadedImage_1.data, uploadedImage_1.cols, uploadedImage_1.rows, uploadedImage_1.step, QImage::Format_Grayscale8);
        QPixmap output = QPixmap::fromImage(qimg);
        ui->thresholdd_output->setPixmap(QPixmap::fromImage(qimg));
        int w= ui->thresholdd_output->width();
        int h= ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
        if (!fileName.isEmpty())
        {

            QImage image(fileName);
            QPixmap pix = QPixmap::fromImage(image);
            ui->Threshold_input->setPixmap(pix);
            int w = ui->Threshold_input->width();
            int h = ui->Threshold_input->height();
            ui->Threshold_input->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

            uploadedImage_1 = cv::imread(fileName.toStdString(),cv::IMREAD_GRAYSCALE);
            spareimage1 = uploadedImage_1;

        }
}


void MainWindow::on_thrshold_option_currentTextChanged(const QString &arg1)
{
    thresholdingOption = ui->thrshold_option->currentText();

}

void MainWindow::on_global_thresholdin_button_clicked()
{

    ui->threshold_value->setVisible(1);
    ui->thres->setVisible(1);
    ui->block_size->setVisible(0);
    ui->blocksize->setVisible(0);
    ui->Threshold_button->setVisible(0);
    if (thresholdingOption == "Optimal Thresholding"){
        Mat globalOptimal=uploadedImage_1;
        int Optimal_threshold=thresholding.optimalThresholding(globalOptimal);
        ui->threshold_value->setText( QString::number( Optimal_threshold ));
        Mat optimal_global= thresholding.globalThresholding(uploadedImage_1,  Optimal_threshold , 255);
        QImage qimg(optimal_global.data, optimal_global.cols, optimal_global.rows, optimal_global.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(output);
        int w = ui->thresholdd_output->width();
        int h = ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

    }
    if (thresholdingOption == "Otsu Thresholding"){
        Mat globalOtsu=uploadedImage_1;
       int otsuThreshold= thresholding.otsuThresholding(globalOtsu);
        ui->threshold_value->setText( QString::number( otsuThreshold ));
        Mat Ostu_global= thresholding.globalThresholding(uploadedImage_1,  otsuThreshold , 255);
        QImage qimg(Ostu_global.data, Ostu_global.cols, Ostu_global.rows, Ostu_global.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(output);
        int w = ui->thresholdd_output->width();
        int h = ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));


    }
    if(thresholdingOption == "Spectural Thresholding"){
        ui->threshold_value->setVisible(0);
        ui->thres->setVisible(0);

        Mat globslcpect_input=uploadedImage_1;
        Mat Global_spectral=thresholding.Global_Spectral(globslcpect_input);
        QImage qimg(Global_spectral.data, Global_spectral.cols, Global_spectral.rows, Global_spectral.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(output);
        int w = ui->thresholdd_output->width();
        int h = ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

    }

}


void MainWindow::on_local_tresholding_button_clicked()
{


    ui->Threshold_button->setVisible(1);
    ui->block_size->setVisible(1);
    ui->blocksize->setVisible(1);
    ui->threshold_value->setVisible(0);
    ui->thres->setVisible(0);



}



void MainWindow::on_Threshold_button_clicked()
{   int LocalBlockSize= ui->block_size->value();
    if (thresholdingOption == "Optimal Thresholding"){

            Mat localOptimal=uploadedImage_1;
            Mat optimal_local=  thresholding.localThresholding(localOptimal,LocalBlockSize,"optimal");
            QImage qimg(optimal_local.data, optimal_local.cols, optimal_local.rows, optimal_local.step, QImage::Format_Grayscale8);
            QPixmap output=QPixmap::fromImage(qimg);
            ui->harris_output->setPixmap(output);
            int w = ui->thresholdd_output->width();
            int h = ui->thresholdd_output->height();
            ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));
    }
    if (thresholdingOption == "Otsu Thresholding"){

        Mat localOtsu=uploadedImage_1;
        Mat otsu_local=  thresholding.localThresholding(localOtsu,LocalBlockSize,"otsu");
        QImage qimg(otsu_local.data, otsu_local.cols, otsu_local.rows, otsu_local.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(output);
        int w = ui->thresholdd_output->width();
        int h = ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

    }
    if(thresholdingOption == "Spectural Thresholding"){
        ui->threshold_value->setVisible(0);
        ui->thres->setVisible(0);

        Mat localspect_input=uploadedImage_1;

        Mat local_spectral=thresholding.Local_Spectral(localspect_input, LocalBlockSize);
        QImage qimg(local_spectral.data, local_spectral.cols, local_spectral.rows, local_spectral.step, QImage::Format_Grayscale8);
        QPixmap output=QPixmap::fromImage(qimg);
        ui->harris_output->setPixmap(output);
        int w = ui->thresholdd_output->width();
        int h = ui->thresholdd_output->height();
        ui->thresholdd_output->setPixmap(output.scaled(w,h,Qt::KeepAspectRatio));

    }


}




//******************************************************************** Tenth Tab(Clustering) ************************************************************************


void MainWindow::on_tab10_browse_clicked()
{

    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Image Files (*.png *.jpg *.bmp)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString fileName= QFileDialog::getOpenFileName(this, tr("Open Image"), "/",
        tr("Image Files (*.png *.jpg *.bmp)"),0, QFileDialog::DontUseNativeDialog);

    if (!fileName.isEmpty())
    {


        uploadedImage_10 = imread(fileName.toStdString(),cv::IMREAD_ANYCOLOR);
        QImage image(fileName);
        QPixmap pix = QPixmap::fromImage(image);
        ui->tab10_img1->setPixmap(pix);
        int w = ui->tab10_img1->width();
        int h = ui->tab10_img1->height();
        ui->tab10_img1->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

    }

}





void MainWindow::on_cluster_options_currentTextChanged(const QString &arg1)
{

    QString option = ui->cluster_options->currentText();

    if (option == "K-Means Clustering"){

        ui->Agg_groupBox->setVisible(0);
        ui->KMeans_groupBox->setVisible(1);

        int k= ui->KMeans_Box->value();
        uploadedImage_10_Segmented= cluster.GetKMeans(uploadedImage_10, k);

        cvtColor(uploadedImage_10_Segmented, uploadedImage_10_Segmented, cv::COLOR_BGR2RGB);
        QImage img2 = QImage((uchar*)uploadedImage_10_Segmented.data, uploadedImage_10_Segmented.cols, uploadedImage_10_Segmented.rows, uploadedImage_10_Segmented.step, QImage::Format_RGB888);

        QPixmap pix2 = QPixmap::fromImage(img2);
        //ui->tab10_img2->setPixmap(pix2);
        int width = ui->tab10_img2->width();
        int height = ui->tab10_img2->height();
        ui->tab10_img2->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));

    }
    if(option=="Region Growing Segmentation"){

        ui->Agg_groupBox->setVisible(0);
        ui->KMeans_groupBox->setVisible(0);
        ui->region_box->setVisible(1);
        ui->points_label->setVisible(1);
        ui->done_button->setVisible(1);
        ui->region_threshold->setVisible(1);


        vector<pair<int, int>> seed_set2 = {{100, 100}};
        int threshold= ui->region_threshold->value();
        cvtColor(uploadedImage_10, uploadedImage_10, cv::COLOR_RGB2GRAY);

         segmented_image = Region_Growing.Region_Growing(uploadedImage_10, seed_set2,threshold,1);
         normalize(segmented_image, segmented_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
          cv::threshold(segmented_image, segmented_image, 128, 255, THRESH_BINARY_INV);

          cvtColor(segmented_image, segmented_image, cv::COLOR_BGR2RGB);
                  QImage img2 = QImage((uchar*)segmented_image.data, segmented_image.cols, segmented_image.rows, segmented_image.step, QImage::Format_RGB888);

                  QPixmap pix2 = QPixmap::fromImage(img2);
                  //ui->tab10_img2->setPixmap(pix2);
                  int width = ui->tab10_img2->width();
                  int height = ui->tab10_img2->height();
                  ui->tab10_img2->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));


    }



    if (option == "Mean Shift Clustering"){



        ui->Agg_groupBox->setVisible(0);
        ui->KMeans_groupBox->setVisible(0);
        ui->meanshift->setVisible(1);
        ui->spatial_BW->setVisible(1);
        ui->color_BW->setVisible(1);
        ui->spatial_val->setVisible(1);
        ui->color_val->setVisible(1);
        ui->meanshift_done->setVisible(1);


        double spatial_val=ui->spatial_val->value();
        double color_val=ui->color_val->value();

        cv::resize(uploadedImage_10, uploadedImage_10, Size(256, 256), 0, 0, 1);
        MeanShift meanShift;
        meanShift.MeanShift_Segmentation(uploadedImage_10, spatial_val, color_val);

        cvtColor(uploadedImage_10, uploadedImage_10, cv::COLOR_BGR2RGB);
                QImage img2 = QImage((uchar*)uploadedImage_10.data, uploadedImage_10.cols, uploadedImage_10.rows, uploadedImage_10.step, QImage::Format_RGB888);

                QPixmap pix2 = QPixmap::fromImage(img2);
                //ui->tab10_img2->setPixmap(pix2);
                int width = ui->tab10_img2->width();
                int height = ui->tab10_img2->height();
                ui->tab10_img2->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));




    }


    if (option == "Agglomerative Clustering"){

        ui->KMeans_groupBox->setVisible(0);
        ui->Agg_groupBox->setVisible(1);

        int bias= ui->KMeans_Box->value();
        int clusters = 30000+bias;
        /*uploadedImage_10_Segmented= cluster.BuildHeirarchy(uploadedImage_10, clusters);

        cvtColor(uploadedImage_10_Segmented, uploadedImage_10_Segmented, cv::COLOR_BGR2RGB);
        QImage img2 = QImage((uchar*)uploadedImage_10_Segmented.data, uploadedImage_10_Segmented.cols, uploadedImage_10_Segmented.rows, uploadedImage_10_Segmented.step, QImage::Format_RGB888);

        QPixmap pix2 = QPixmap::fromImage(img2);
        int width = ui->tab10_img2->width();
        int height = ui->tab10_img2->height();
        ui->tab10_img2->setPixmap(pix2.scaled(width,height,Qt::KeepAspectRatio));*/


    }

}




void MainWindow::on_KMeans_Box_valueChanged(int arg1)
{
    on_cluster_options_currentTextChanged("NULL");
}


void MainWindow::on_done_button_clicked()
{

    ui->region_box->setVisible(0);
    ui->points_label->setVisible(0);
    ui->done_button->setVisible(0);
    ui->region_threshold->setVisible(0);


}


void MainWindow::on_region_threshold_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");

}


void MainWindow::on_meanshift_done_clicked()
{
    ui->meanshift->setVisible(0);
    ui->spatial_BW->setVisible(0);
    ui->color_BW->setVisible(0);
    ui->spatial_val->setVisible(0);
    ui->color_val->setVisible(0);
    ui->meanshift_done->setVisible(0);

}



void MainWindow::on_spatial_val_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");

}


void MainWindow::on_color_val_valueChanged(double arg1)
{
    on_ObjectBox_currentTextChanged("test");

}

