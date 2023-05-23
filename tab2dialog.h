#ifndef TAB2DIALOG_H
#define TAB2DIALOG_H

#include <QDialog>

namespace Ui {
class Tab2Dialog;
}

class Tab2Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Tab2Dialog(QWidget *parent = nullptr);
    ~Tab2Dialog();
    QString Get_Norm_Number;

private slots:
    void on_pushButton_clicked();


private:
    Ui::Tab2Dialog *ui;

};

#endif // TAB2DIALOG_H
