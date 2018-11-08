#ifndef BPNN_H
#define BPNN_H
#include <iostream>
#include <vector>
#include <QVector>

namespace YHL {

    class BPNN {
        static constexpr int layer = 3;
        static constexpr double rate = 0.24;
        static constexpr double N = 0.8;
        static constexpr double M = 0.2;
        using matrix = std::vector< std::vector<double> >;
    private:
        // layer - 1 层权值矩阵
        matrix weights[layer - 1];
        matrix before[layer - 1];
        //
        matrix dataSet;
        matrix answers;
        std::vector<double> target;
        // 两层之间的向量
        std::vector<double> output[layer];
        std::vector<double> delta[layer - 1];
        std::vector<double> threshold[layer - 1];

        const int inputSize;
        const int hideSize;
        const int outputSize;

        std::pair< QVector<double>, QVector<double> > efficiency;

    private:
        void initWeights();
        void initOutput();
        void initDelta();
        void initThreshold();
        void initBefore();
        void initDataSet();

        double getError();
        void forwardDrive();
        void backPropagate();
        int chooseBest() const;

        // 收集角度版
        void loadSet(const int sampleCnt);

    public:
        BPNN(const int l, const int m, const int r);
        ~BPNN();
        void train();
        void trainMyself(const int sampleCnt);
        int recognize(const std::vector<double>& input);
        void loadFile(const std::string& fileName);
        const std::pair< QVector<double>, QVector<double> >& getEfficiency() const;
    };
}

#endif // BPNN_H
