#ifndef CLUSTERS_H
#define CLUSTERS_H
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <list>
#include <unordered_map>
#include <set>
#include <QPoint>

namespace YHL {

    struct point {
        double x, y;
        point(const double _x, const double _y);
        bool operator==(const point& rhs) const;
    };
    double getDistance(const point& a, const point& b);

    class Hierarchical {

        using oneCluster = std::vector<point>;
        using ansType = std::unordered_map<int, std::list<int> >;
        struct Node {
            int end, next;
            Node(const int _end, const int _next)
                : end(_end), next(_next) {}
        };
    private:
        int k = 0;
        int len = 0;
        std::string path;

        oneCluster dataSet;
        std::vector<Node> edges;
        std::vector<int> head;
        std::vector< std::vector<int> > maps;

        std::vector<int> color;
        std::unordered_map<int, std::list<int> > clusters;
        std::unordered_map<int, int> others;

    private:
        void DFS(const int u);
        void BFS(const int u);
        void initDis(const double threshold);
        void clear();
        void repair();

        const ansType getClusters(const double threshold);
        const std::unordered_map<int, int>& getOthers();
        const oneCluster& getDataSet() const;
    public:
        const std::vector< std::vector<QPoint> > getCluster(const double threshold);
        void load(const std::vector<QPoint>& travels);
        void readData(const std::string& _path = "hierarchical.txt");
    };
}



#endif // CLUSTERS_H
