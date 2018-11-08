#include "clusters.h"
#include <QDebug>
#include <functional>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <queue>
#include <assert.h>
#include <QMessageBox>
#include "scopeguard.h"
#include <QDebug>

// 宽度优先搜索
void YHL::Hierarchical::BFS (const int u) {
    std::queue<int> Q;
    Q.push (u);
    while(!Q.empty ()) {
        auto front = Q.front ();
        Q.pop ();
        for(int k = head[front];k not_eq -1; k = edges[k].next) {
            auto v = edges[k].end;
            if(color[v] == 0) {
                color[v] = color[u];
                clusters[color[v]].emplace_back(v);
                Q.push (v);
            }
        }
    }
}

// 深度优先搜索
void YHL::Hierarchical::DFS(const int u)
{
//    for(int k = head[u];k not_eq -1; k = edges[k].next) {
//        auto v = edges[k].end;
//        if(color[v] == 0) {
//            color[v] = color[u];
//            clusters[color[v]].emplace_back(v);
//            DFS(v);
//        }
//    }
    for(int i = 0;i < len; ++i) {
        if(maps[u][i] == 1 and color[i] == 0) {
            color[i] = color[u];
            clusters[color[i]].emplace_back(i);
            DFS(i);
        }
    }
}

// 初始化矩阵
void YHL::Hierarchical::initDis(const double threshold) {
    this->head.assign (len, -1);
    this->color.assign (len, 0);

    this->maps.assign (len, std::vector<int>(len, 0));
    for(int i = 0;i < len; ++i) {
        for(int j = 0;j < len; ++j) {  // j < i
            if(i == j) continue;
            auto distance = getDistance (dataSet[i], dataSet[j]);
            if(distance - threshold <= 0) {
               this->edges.emplace_back(j, head[i]);
               head[i] = k++;
               this->maps[i][j] = 1;
            }
        }
    }
}

void YHL::Hierarchical::clear()
{
    this->len = 0;
    this->dataSet.clear ();
    this->clusters.clear ();
    this->others.clear ();
    this->head.clear ();
    this->color.clear ();
    this->edges.clear ();
    this->k = 0;
}

void YHL::Hierarchical::repair()
{
    auto it = clusters.begin ();
    while(it not_eq clusters.end ()) {
        qDebug() << "簇大小" << it->second.size ();
        int scale = it->second.size ();
        if(scale > 5 and scale < 40) {
            auto min = 1e12;
            int index = 0;
            for(const auto& cur : it->second) {
                for(const auto& l : this->clusters) {
                    if(it->first == l.first)
                        continue;
                    for(const auto& r : l.second) {
                        auto distance = getDistance (this->dataSet[r], this->dataSet[cur]);
                        if(distance < min) {
                            min = distance;
                            index = l.first;
                        }
                    }
                }
            }
            qDebug() << "min  :  " << min;
            double x_max = -1e12, x_min = 1e12, y_max = -1e12, y_min = 1e12;
            for(const auto& l : it->second) {
                double x = this->dataSet[l].x;
                double y = this->dataSet[l].y;
                if(x > x_max) x_max = x;
                if(x < x_min) x_min = x;
                if(y > y_max) y_max = y;
                if(y < y_min) y_min = y;
            }
            auto threshold = std::max (x_max - x_min, y_max - y_min);
            qDebug() << "threshild  :  " << threshold;
            if(min < threshold) {
                for(const auto& l : it->second)
                    this->clusters[index].emplace_back (l);
            } else {
                for(auto r : it->second)
                    others.emplace(r, 1);
            }
            it = clusters.erase (it);
        }
        else if(scale <= 5) it = clusters.erase (it);
        else ++it;
    }
}

const YHL::Hierarchical::ansType YHL::Hierarchical::getClusters
        (const double threshold) {
    // 初始化邻接表
    this->initDis (threshold);
    int cnt = 0;
    for(int i = 0;i < len; ++i) {
        if(color[i] == 0) {
            color[i] = ++cnt;  // 开始染色
            clusters[cnt].emplace_back(i);
            BFS(i);  // DFS 也可以
        }
    }
    //后期处理离群点
    this->repair ();
    // 为下次聚类判断是否要读文件做铺垫
    this->len = 0;
    return this->clusters;
}

const std::unordered_map<int, int>& YHL::Hierarchical::getOthers()
{
    return this->others;
}

void YHL::Hierarchical::load(const std::vector<QPoint> &travels)
{
    this->len = travels.size ();
    for(const auto& it : travels) {
        this->dataSet.emplace_back(it.x (), it.y ());
    }
}

// 读取数据
void YHL::Hierarchical::readData(const std::string &_path)
{
    this->path = _path;
    this->clear();

    std::ifstream in(path.c_str());
    ON_SCOPE_EXIT([&]{ in.close(); });
    in >> this->len;
    double x = 0.00, y = 0.00;
    for(int i = 0;i < this->len; ++i) {
        in >> x >> y;
        this->dataSet.emplace_back(x, y);
    }
}

const YHL::Hierarchical::oneCluster &YHL::Hierarchical::getDataSet() const
{
    return this->dataSet;
}

const std::vector<std::vector<QPoint> >
    YHL::Hierarchical::getCluster(const double threshold)
{
    std::vector< std::vector<QPoint> > ans;
    auto one = this->getClusters (threshold);
    qDebug () << "cluster Size  :  " << this->clusters.size ();
    for(const auto& it : this->clusters) {
        std::vector<QPoint> a;
        for(const auto r : it.second) {
            a.emplace_back(this->dataSet[r].x, this->dataSet[r].y);
        }
        ans.emplace_back(a);
    }
    std::sort(ans.begin (), ans.end (), [&](std::vector<QPoint>& a,std::vector<QPoint>& b){
        double lhs = 0.00, rhs = 0.00;
        int len1 = a.size ();
        for(int i = 0;i < len1; ++i)
            lhs += a[i].x ();
        lhs /= len1;
        int len2 = b.size ();
        for(int i = 0;i < len2; ++i)
            rhs += b[i].x ();
        rhs /= len2;
        return lhs < rhs;
    });
    return ans;
}

double YHL::getDistance(const YHL::point &a, const YHL::point &b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));}

YHL::point::point(const double _x, const double _y)
    : x(_x), y(_y){}
bool YHL::point::operator==(const YHL::point &rhs) const {
    return (this->x - rhs.x) < 1e-3 and (this->y - rhs.y) < 1e-3;
}
