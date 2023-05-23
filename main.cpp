#include "mainwindow.h"
#include <QApplication>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;



int main(int argc, char *argv[])

{

    //Mat originalimage;
    //originalimage= imread("/image.png", IMREAD_GRAYSCALE);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();

}
