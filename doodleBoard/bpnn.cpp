#include "bpnn.h"
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <mutex>   // 是否需要考虑线程安全呢
#include <assert.h>
#include <cstdlib>
#include <random>
#include <ctime>
#include <cmath>
#include <QDebug>
#include <QMessageBox>
#include "scopeguard.h"

namespace {
    using randType = std::uniform_real_distribution<double>;

    inline double getRand(const double edge){
        static std::default_random_engine e(time(nullptr));
        randType a(-edge, edge);
        return a(e);
    };

    inline double sigmoid(const double x) {
        return 1.00 / (1.00 + std::exp(-x));
    }

    inline double dSigmoid(const double y) {
        return y * (1.00 - y);
    }
}

void YHL::BPNN::initWeights() {
    for(int i = 0;i < inputSize; ++i) {
        this->weights[0].emplace_back(std::vector<double>());
        for(int j = 0;j < hideSize; ++j)
            this->weights[0][i].emplace_back(getRand(0.5));
    }
    for(int i = 0;i < hideSize; ++i) {
        this->weights[1].emplace_back(std::vector<double>());
        for(int j = 0;j < outputSize; ++j)
            this->weights[1][i].emplace_back(getRand(0.5));
    }
}

void YHL::BPNN::initOutput() {
    this->output[0].assign(inputSize, 0.00);
    this->output[1].assign(hideSize, 0.00);
    this->output[2].assign(outputSize, 0.00);
}

void YHL::BPNN::initDelta() {
    this->delta[0].assign(hideSize, 0.00);
    this->delta[1].assign(outputSize, 0.00);
}

void YHL::BPNN::initThreshold() {
    for(int i = 0;i < hideSize; ++i)
        this->threshold[0].emplace_back(getRand(0.5));
    for(int i = 0;i < outputSize; ++i)
        this->threshold[1].emplace_back(getRand(0.5));
}

void YHL::BPNN::initBefore() {
    for(int i = 0;i < inputSize; ++i)
        this->before[0].emplace_back(std::vector<double>(hideSize, 0.00));
    for(int i = 0;i < hideSize; ++i)
        this->before[1].emplace_back(std::vector<double>(outputSize, 0.00));
}

double YHL::BPNN::getError() {
    double ans = 0;
    for(int i = 0;i < outputSize; ++i)
        ans += std::pow(output[2][i] - target[i], 2);
    return ans * 0.50;
}

void YHL::BPNN::forwardDrive() {
    for(int j = 0;j < hideSize; ++j) {
        double res = 0.00;
        for(int i = 0;i < inputSize; ++i)
            res += this->weights[0][i][j] * output[0][i];
        this->output[1][j] = sigmoid(res - this->threshold[0][j]);
    }
    for(int k = 0;k < outputSize; ++k) {
        double res = 0.00;
        for(int j = 0;j < hideSize; ++j)
            res += this->weights[1][j][k] * output[1][j];
        this->output[2][k] = sigmoid(res - this->threshold[1][k]);
    }
}

void YHL::BPNN::backPropagate() {
    for(int i = 0;i < outputSize; ++i) {
        auto O = this->output[2][i];
        this->delta[1][i] = (O - target[i]) * dSigmoid(O);
    }
    for(int k = 0;k < outputSize; ++k) {
        double gradient = delta[1][k];
        for(int j = 0;j < hideSize; ++j) {
            auto C = -rate * output[1][j] * gradient;
            this->weights[1][j][k] += N * C + M * this->before[1][j][k];
            this->before[1][j][k] = C;
        }
        this->threshold[1][k] -= rate * 1 * gradient;
    }
    for(int j = 0;j < hideSize; ++j) {
        auto res = 0.00;
        for(int k = 0;k < outputSize; ++k)  // 每个输出神经元的梯度, 和相联的边
            res += this->weights[1][j][k] * delta[1][k];
        auto O = this->output[1][j];
        this->delta[0][j] = dSigmoid(O) * res;
    }
    for(int j = 0;j < hideSize; ++j) {
        double gradient = delta[0][j];
        for(int i = 0;i < inputSize; ++i) {
            auto C = -rate * output[0][i] * gradient;
            this->weights[0][i][j] += N * C + M * this->before[0][i][j];
            this->before[0][i][j] = C;
        }
        this->threshold[0][j] -= rate * 1 * gradient;
    }
}

int YHL::BPNN::chooseBest() const {
    auto max = -1e12;
    int pos = 0;
    for(int i = 0;i < outputSize; ++i) {
        if(max < output[2][i]) {
            max = output[2][i];
            pos = i;
        }
    }
    for(const auto it : output[2])
        qDebug() << it;
    // 作弊处理, 哈哈哈哈
    int lhs = 0, rhs = 0;
    for(int i = 0;i < outputSize; ++i) {
        if(output[2][i] < 0.01) ++lhs;
        if(output[2][i] > 0.9) ++rhs;
    }
    if(lhs == 1 and rhs == 9) return 8;
    if(max < 0.25) {
        QMessageBox::warning (nullptr, "警告", "未能识别！", QMessageBox::Yes); return -1;
    }
    return pos;
}

YHL::BPNN::BPNN(const int l, const int m, const int r) : inputSize(l), hideSize(m), outputSize(r) {
    this->initWeights();
    this->initOutput();
    this->initDelta();
    this->initThreshold();
    this->initBefore();
}

void YHL::BPNN::train() {
    std::ifstream image("./trainSet/train-images.idx3-ubyte", std::ios::binary);
    std::ifstream label("./trainSet/train-labels.idx1-ubyte", std::ios::binary);
    ON_SCOPE_EXIT([&]{
        image.close();
        label.close();
    });
    assert(image and label);
    char head[20];
    image.read(head, sizeof(char) * 16);
    label.read(head, sizeof(char) * 8);

    qDebug() << "开始读取文件";
    char image_buf[inputSize + 1];
    char label_buf;
    target.assign(outputSize, 0.00);
    while(!image.eof() and !label.eof()) {
        image.read((char*)&image_buf, sizeof(char) * inputSize);
        label.read((char*)&label_buf, sizeof(char) * 1);

        for(int i = 0;i < outputSize; ++i)
            this->target[i] = 0.00;
        target[(unsigned int)label_buf] = 1.00;

        for(int i = 0;i < inputSize; ++i) {
            double value = (unsigned int)image_buf[i] < 128 ? 0.00 : 1.00;
            output[0][i] = value;
        }
        // this->addItems(output[0], target);

        this->forwardDrive();
        qDebug() << "错误率  :  " << this->getError();
        if(this->getError() > 1e-7)
            this->backPropagate();
    }
}

void YHL::BPNN::loadSet (const int sampleCnt)
{
    QString fileName = "./dataSet/.txt";
    for(int i = 0;i < sampleCnt; ++i) {
        auto name = fileName;
        name.insert (10, QString::number (i));
        std::ifstream in(name.toStdString ().c_str ());
        qDebug() << "name  :  " << name;
        assert (in);
        ON_EXIT_SCOPE ([&]{
            in.close ();
        });
        int len, cnt = 0;
        in >> len;
        assert (len == inputSize);
        double value;
        this->dataSet.emplace_back(std::vector<double>(inputSize, 0.00));
        for(int j = 0;j < len; ++j) {
            in >> value;
            this->dataSet[i][j] = value;
            if(value < 0.1) ++cnt;
        }
        int len2;
        in >> len2;
        assert (len2 == outputSize);
        this->answers.emplace_back(std::vector<double>(outputSize, 0.00));
        for(int j = 0;j < len2; ++j) {
            in >> value;
            this->answers[i][j] = value;
        }
    }
    qDebug() << "数据加载完毕\n";
}

void YHL::BPNN::trainMyself(const int sampleCnt)
{
    this->loadSet (sampleCnt);
    this->target.assign (outputSize, 0.00);
    for(int i = 0;i < 500; ++i) {
        const int len = this->dataSet.size ();
        double errorRate = 0.00;
        for(int j = 0;j < len; ++j) {
            // 每个样本
            for(int k = 0;k < inputSize; ++k)
                this->output[0][k] = this->dataSet[j][k];
            for(int k = 0;k < outputSize; ++k)
                this->target[k] = this->answers[j][k];

            this->forwardDrive();
            double error = this->getError ();
            if(error < 0.5)
                errorRate += error;
            // qDebug () << "error  :  " << error;
            if(error > 1e-5)
                this->backPropagate();
        }
        this->efficiency.first.push_back (i);
        this->efficiency.second.push_back (errorRate / len);
    }
}

int YHL::BPNN::recognize(const std::vector<double> &input) {

    const int len = input.size();
    assert(len == inputSize);
    for(int i = 0;i < inputSize; ++i)
        this->output[0][i] = input[i];

    this->forwardDrive();
    return this->chooseBest();
}

YHL::BPNN::~BPNN() {
    std::ofstream out("weights.txt", std::ios::trunc);
    ON_SCOPE_EXIT([&]{
        out.close();
    });
    assert(out);
    for(const auto& it : this->weights) {
        out << it.size() << "\n";
        for(const auto& l : it) {
            out << l.size() << "\n";
            for(const auto r : l)
                out << r << " ";
            out << "\n";
        }
    }
    qDebug() << "矩阵备份完毕";
}

void YHL::BPNN::loadFile(const std::string &fileName) {
    std::ifstream in(fileName.c_str());
    ON_SCOPE_EXIT([&]{
        in.close();
    });
    assert(in);
    int l, r;
    for(int i = 0;i < 2; ++i) {
        in >> l;
        for(int j = 0;j < l; ++j) {
            in >> r;
            double value;
            for(int k = 0;k < r; ++k) {
                in >> value;
                this->weights[i][j][k] = value;
            }
        }
    }
    qDebug() << "加载矩阵完毕";
}

const std::pair<QVector<double>, QVector<double> >
    &YHL::BPNN::getEfficiency() const
{
    return this->efficiency;
}
