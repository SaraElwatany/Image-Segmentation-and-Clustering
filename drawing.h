#ifndef DRAWING_H
#define DRAWING_H


#include <QWidget>
#include <QPainter>
#include <QImage>
#include <QMouseEvent>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


class ImageWidget : public QWidget {
public:
    ImageWidget(const cv::Mat& image, QWidget* parent = nullptr) : QWidget(parent), image(image) {
        setFixedSize(image.cols, image.rows);
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        QPainter painter(this);
        QImage qimage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
        painter.drawImage(0, 0, qimage);
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            emit leftMouseButtonPressed(event->pos());
        }
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if (event->buttons() & Qt::LeftButton) {
            emit leftMouseButtonMoved(event->pos());
        }
    }

private:
    cv::Mat image;

signals:
    void leftMouseButtonPressed(const QPoint& pos);
    void leftMouseButtonMoved(const QPoint& pos);
};

#endif // DRAWING_H
