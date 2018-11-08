#ifndef DOODLEBOARD_H
#define DOODLEBOARD_H
#include "qcustomplot.h"
#include <QMainWindow>
#include <QMouseEvent>
#include <memory>
#include <number.h>
#include <unordered_map>
#include <QTimerEvent>
#include <vector>
#include <thread>
#include <mutex>

namespace Ui {
class doodleBoard;
}
namespace YHL {
class BPNN;
class Hierarchical;
}

class doodleBoard : public QMainWindow
{
    Q_OBJECT
public:
    explicit doodleBoard(QWidget *parent = nullptr);
    ~doodleBoard();

private:
    void initSave ();

protected:
    void paintEvent(QPaintEvent*);
    void mousePressEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent*);
    void mouseReleaseEvent(QMouseEvent*);
    void closeEvent(QCloseEvent*);

private:
    std::vector<QPoint> reSort(const std::vector<QPoint>& oneCluster);
    bool recodeTravel(std::vector<QPoint>& oneCluster);
    QPoint getCenter(const std::vector<QPoint>& oneCluster);
    QImage getDevision(QImage image);
    void runBP();

private slots:
    void on_close_button_clicked();

    void on_clear_button_clicked();

    void on_recognize_clicked();

    void on_save_button_clicked();

    void on_train_button_clicked();

    void on_set_button_clicked();

private:
    // 神经网络相关
    std::unique_ptr<YHL::BPNN> bpnn;
    QImage ultimate;
    bool trained = false;
    bool being_trained = false;

    // 线程辅助
    std::thread trainThread;
    std::thread tips;
    std::mutex mtx;

    // 绘图相关
    Ui::doodleBoard *ui;
    QPixmap pix;
    QPoint lastPoint;
    QPoint endPoint;
    QPoint goWith;
    std::list<QCustomPlot*> litters;

    // 收集角度相关
    int inputSize = 40; // 输入层维度
    int outputSize = 10;// 输出层维度 0 - 9
    int sampleCnt;      // 当前存的样本个数
    std::vector<QPoint> travels;      // 轨迹
    std::vector<double> angles;       // 角度
    std::vector<double> input;        // 真正的输入
    std::unique_ptr<number> numDialog;// 保存样本的窗口
    std::unordered_map<int, int> book;// 记录每个数字的图片保存到哪儿了
    struct pair {
        int index;
        double feature;
        pair(const int _index, const double _feature)
            : index(_index), feature(_feature){}
    };
    std::vector<pair> getFilter();
};

#endif // DOODLEBOARD_H
