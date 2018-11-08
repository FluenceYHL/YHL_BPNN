#include "doodleboard.h"
#include "ui_doodleboard.h"
#include <QPainter>
#include <QPixmap>
#include <QDebug>
#include <QImage>
#include <QMatrix>
#include <QImageReader>
#include <QMessageBox>
#include <fstream>
#include <QTimer>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include "scopeguard.h"
#include "bpnn.h"
#include "clusters.h"

namespace {

    double getDiatance(const QPoint &lhs, const QPoint &rhs)
    {
        double delta_x = lhs.x () - rhs.x ();
        double delta_y = lhs.y () - rhs.y ();
        return std::sqrt (delta_x * delta_x + delta_y * delta_y);
    }

    double getCos(const double a, const double b, const double c)
    {
        double denominator = a * a + b * b - c * c;
        double molecule = 2 * a * b;
        // *180.00 归一化处理
        return std::acos (denominator / molecule) / M_PI;
    }

    bool is_same(const QPoint& a, const QPoint& b) {
        return std::fabs (a.x () - b.x ()) < 1e-4 and
               std::fabs (a.y () - b.y ()) < 1e-4;
    }

    std::vector<double> getVector(const QImage& image) {
        std::vector<double> collect;
        QRgb *oneLine = nullptr;
        const int height = image.height ();
        for(int i = 0;i < height; ++i) {
            oneLine = (QRgb*)image.scanLine (i);
            const int width = image.width ();
            for(int j = 0;j < width; ++j) {
                if(qRed(oneLine[j]) == 0 || qGreen (oneLine[j] == 0 || qBlue (oneLine[j]) == 0))
                    collect.emplace_back(1);
                else
                    collect.emplace_back(0);
            }
        }
        return collect;
    }
}

namespace {

    void logCall(const QString& message) {
        QMessageBox::warning (nullptr, "警告!", message, QMessageBox::Yes);
    }

    QColor getColor() {
        static QList<QColor> colors{ Qt::red, Qt::black, Qt::blue, Qt::yellow, Qt::green, Qt::gray };
        if(!colors.empty ()) {
            auto one = colors.front ();
            colors.pop_front ();
            return one;
        }
        int a = rand() % 255, b = rand() % 255, c = rand() % 255;
        return qRgb(a, b, c);
    }

}

doodleBoard::doodleBoard(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::doodleBoard)
{
    ui->setupUi(this);
    this->setWindowTitle ("YHL 涂鸦板");

    QFont font;
    font.setPointSize (100);
    ui->result->setFont (font);
    ui->result->setStyleSheet("background-color:#DDDDDD");
    ui->result->setAlignment (Qt::AlignCenter);
    ui->result->setWordWrap (true);

    pix = QPixmap(600, this->height ());
    pix.fill (Qt::white);
    this->initSave ();

    srand(static_cast<unsigned int>(time(nullptr)));
}

doodleBoard::~doodleBoard()
{
    delete ui;
    ui = nullptr;
    std::ofstream out("./saveCnt.txt", std::ios::trunc);
    YHL::ON_SCOPE_EXIT([&]{ out.close (); });
    out << this->sampleCnt << "\n";
    out << this->book.size () << "\n";
    for(const auto& it : this->book) {
        out << it.first << " " << it.second << "\n";
    }
}

void doodleBoard::initSave()
{
    this->numDialog = std::make_unique<number>();
    connect (this->numDialog.get (), &number::send_number, [&](const int arg){
        QString path = "./dataSet/.txt";
        path.insert (10, QString::number (this->sampleCnt));
         ++this->sampleCnt;
        qDebug() << "path  :  " << path;
        // 保存训练集
        std::ofstream out(path.toStdString ().c_str (), std::ios::trunc);
        YHL::ON_SCOPE_EXIT([&]{
            out.close ();
        });
        out << this->inputSize << "\n";
        for(const auto it : this->input)
            out << it << "\n";
        out << this->outputSize << "\n";
        for(int i = 0;i < outputSize; ++i) {
            if(i == arg) out << "1.00" << " ";
            else out << "0.00" << " ";
        }
        // 保存图片
        QString picturePath = "./dataSet//.png";
        picturePath.insert (10, QString::number (arg));
        picturePath.insert (12, QString::number (this->book[arg]++));
        pix.save (picturePath);
    });
    // 读取文件
    std::ifstream in("./saveCnt.txt");
    YHL::ON_SCOPE_EXIT([&]{ in.close (); });
    in >> this->sampleCnt;
    int Size;
    in >> Size;
    int obj, times;
    for(int i = 0;i < Size; ++i) {
        in >> obj >> times;
        this->book.emplace (obj, times);
    }
}

void doodleBoard::paintEvent(QPaintEvent *)
{
    QPainter help(&pix);
    help.setPen (QPen(Qt::black, 15));
    help.drawLine (lastPoint, endPoint);
    lastPoint = endPoint;
    QPainter painter(this);
    painter.drawPixmap (0, 0, pix);
}

void doodleBoard::mousePressEvent(QMouseEvent *e)
{
    if(e->button () == Qt::LeftButton) {
        lastPoint = e->pos ();
    }
    endPoint = lastPoint;
}

void doodleBoard::mouseMoveEvent(QMouseEvent *e)
{
    if(e->buttons () & Qt::LeftButton) {
        endPoint = e->pos ();
        static long long cnt = 0;
        // 调整收集轨迹点的密度
        if(++cnt % 3 == 0) {
            this->travels.emplace_back(endPoint);
            if(cnt >= std::numeric_limits<long long>::max ()) cnt = 0;
        }
        update ();
    }
    this->goWith = e->pos ();
}

void doodleBoard::mouseReleaseEvent(QMouseEvent *e)
{
    if(e->button () == Qt::LeftButton) {
        endPoint = e->pos (); update ();
        this->runBP ();
    }
}

void doodleBoard::closeEvent(QCloseEvent *)
{
    this->close ();
}

void doodleBoard::on_close_button_clicked()
{
    this->close ();
}

void doodleBoard::on_clear_button_clicked()
{
    pix.fill (Qt::white);
    update ();
    lastPoint.setX (0); lastPoint.setY (0);
    endPoint.setX (0); endPoint.setY (0);
    this->travels.clear ();
    this->angles.clear ();
    this->input.clear ();
    ui->result->clear ();
}

void doodleBoard::on_recognize_clicked()
{
    this->runBP ();
}

void doodleBoard::on_save_button_clicked()
{
    assert (this->numDialog);
    numDialog->show ();
    this->recodeTravel (this->travels);
}

void doodleBoard::on_set_button_clicked()
{
    auto one = this->bpnn->getEfficiency ();
    QCustomPlot *gragh = new QCustomPlot();
    this->litters.emplace_back (gragh);
    gragh->addGraph (gragh->xAxis, gragh->yAxis);
    gragh->graph (0)->setName ("随着迭代次数增加，误差函数的变化");
    gragh->setInteractions (QCP::iRangeDrag | QCP::iRangeZoom);
    gragh->legend->setVisible (true);
    gragh->xAxis->setRange (0, 500);
    gragh->yAxis->setRange (0, 0.5);
    gragh->xAxis->setLabel ("迭代次数");
    gragh->yAxis->setLabel ("损失函数");
    gragh->show ();

    auto curGragh = gragh->addGraph ();
    curGragh->setPen (QPen(getColor (), 6));
    curGragh->setData (one.first, one.second, false);
    curGragh->setName ("损失函数曲线");
    gragh->replot ();
    gragh->savePng ("efficiency.png");
}

void doodleBoard::on_train_button_clicked()
{
    if(this->being_trained == true) {
        QMessageBox::information (this, "警告", "正在训练中！", QMessageBox::Yes); return;
    }
    auto one = std::make_unique<YHL::BPNN>(40, 15, 10);
    assert (one);
    one.swap(this->bpnn);
    this->trained = false;
    this->being_trained = true;
    this->trainThread = std::thread([&]{
        this->bpnn->trainMyself (this->sampleCnt);
        this->trained = true;
        this->being_trained = false;
    });
    this->trainThread.detach ();
}

void doodleBoard::runBP()
{
    if(!this->bpnn) {
        QMessageBox::warning (this, "警告", "网络尚未训练 !", QMessageBox::Yes); return;
    }
    if(this->being_trained == true) {
        QMessageBox::warning (this, "警告", "网络正在训练中", QMessageBox::Yes); return;
    }
    std::lock_guard<std::mutex> lck(mtx);

    YHL::Hierarchical one;
    one.load(this->travels);
    auto ans = one.getCluster (15);
    qDebug() << "聚类个数  :  " << ans.size ();  // 顺序问题，可以靠排序来解决,计算 x 均值
    int screen = 0;
    QString oneFlock = "";
    for(const auto& it : ans) {  // reSort (it)
        auto sorted = reSort (it);
        if(this->recodeTravel (sorted)) {   // 邻接表的锅.....
            auto result = this->bpnn->recognize (this->input);
            screen = screen * 10 + result;
            oneFlock.append (QString::number (result));
        } else {
            this->travels.clear ();
            pix.fill (Qt::white);
            return;
        }
    }
    ui->result->setText (oneFlock);
}

std::vector<QPoint> doodleBoard::reSort(const std::vector<QPoint> &oneCluster)
{
    std::vector<QPoint> one;
    for(const auto& it : travels) {
        for(const auto& r : oneCluster) {
            if(is_same (it, r))
                one.emplace_back (it);
        }
    }
    return one;
}

QImage doodleBoard::getDevision(QImage image)
{
    int left = 1e5;
    int right = 0;
    int ceil = 1e5;
    int floor = 0;
    for(const auto& it : this->travels) {
        if(left > it.x ()) left = it.x ();
        if(right < it.x ()) right = it.x ();
        if(ceil > it.y ()) ceil = it.y ();
        if(floor < it.y ()) floor = it.y ();
    }
    qDebug() << "left  :  " << left;
    qDebug() << "right :  " << right;
    qDebug() << "ceil  :  " << ceil;
    qDebug() << "floor :  " << floor;
    return image.copy (std::max(0, left - 80), std::min(380, ceil - 80),
                       std::min(floor - ceil + 160, 540 - ceil),
                       std::min(floor - ceil + 160, 540 - ceil));
}









































// 如果收集距离,就需要放大和缩小,以及高矮胖瘦的问题,可以很好地解决倾斜的问题,但是 3, 8, 5 这些....
// 或者是比值,从起点到某个点的长度,但是放大和缩小也不容易
// 如果收集角度,讲真的,很匹配,但是最大的问题就是倾斜......解决办法是三个点确定一个角度,余弦定理
// 以上两种思路都不行,还是矩阵最理想; 最后再试试新 BP 在轨迹上的效果
bool doodleBoard::recodeTravel(std::vector<QPoint>& oneCluster)
{
    YHL::ON_SCOPE_EXIT([&]{
        this->angles.clear ();
    });
    auto &lhs = oneCluster[0];
    auto rhs = this->getCenter (oneCluster);
    double c = getDiatance (lhs, rhs);

    int travelSize = oneCluster.size ();
    for(int i = 1;i < travelSize; ++i) {
        if(oneCluster[i] == rhs) // 如果和第二个点重合 std::fabs < 1e-4 后期优化
            continue;
        double a = getDiatance (lhs, oneCluster[i]);
        double b = getDiatance (rhs, oneCluster[i]);
        double angle = getCos (a, b, c);
        this->angles.emplace_back (angle);
    }
    // 可以增大收集密度，同时要限制获取的特征长度
    int vecSize = angles.size ();
    if(vecSize < inputSize) {
        QMessageBox::warning (this, "警告", "数据采集不足, 请放慢书写速度", QMessageBox::Yes);
        return false;
    }
    auto filter = this->getFilter ();
    this->input.clear ();
    for(const auto it : filter)
        this->input.emplace_back (it.feature);
    return true;
}

std::vector<doodleBoard::pair> doodleBoard::getFilter()
{
    // 从中随机选择 inputSize 个特征值,而且要保证有序
    int vecSize = this->angles.size ();
    std::vector<pair> filter;  // 最后要注意 -1 这个参数的插入
    std::vector<int> book(vecSize, 0);
    for(int i = 0;i < inputSize; ++i) {
        auto cur = rand() % vecSize;
        while(book[cur] not_eq 0)
            cur = rand() % vecSize;
        book[cur] = 1;
        filter.emplace_back(cur, angles[cur]);
    }
    std::sort (filter.begin (), filter.end (), [&](const pair& a, const pair& b){
        return a.index < b.index;
    });
    return filter;
}

QPoint doodleBoard::getCenter(const std::vector<QPoint>& oneCluster)
{
    double ans_x = 0.00, ans_y = 0.00;
    for(const auto& it : oneCluster) {
        ans_x += it.x ();
        ans_y += it.y ();
    }
    int len = oneCluster.size ();
    return QPoint(ans_x / len, ans_y / len);
}


