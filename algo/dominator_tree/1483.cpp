#include <vector>

using namespace std;

class TreeAncestor
{
public:
    TreeAncestor(int n, vector<int> &parent)
    {
        g = vector<vector<int>>(n);
        for (int node = 1; node < n; ++node)
            g[parent[node]].push_back(node);
        id = vector<int>(n, -1);
        nodeid = vector<int>(n, -1);
        level_dfnids = vector<vector<int>>();
        node_level = vector<int>(n, -1);
        int dfnid = 0;
        dfs(0, dfnid, 0);
    }

    int getKthAncestor(int node, int k)
    {
        int level = node_level[node];
        int ancestor_level = level - k;
        if (ancestor_level < 0)
            return -1;
        int dfnid = id[node];
        // 大于等于 dfnid 的最小 it
        auto it = lower_bound(level_dfnids[ancestor_level].begin(), level_dfnids[ancestor_level].end(), dfnid);
        int ancestor_dfnid = *(--it);
        return nodeid[ancestor_dfnid];
    }

private:
    vector<int> id;                   // nodeid -> dfnid
    vector<int> nodeid;               // dfnid -> nodeid
    vector<int> node_level;           // nodeid -> level
    vector<vector<int>> level_dfnids; // level -> dfnids
    vector<vector<int>> g;            // 邻接表

    void dfs(int node, int &dfnid, int level)
    {
        id[node] = dfnid;
        nodeid[dfnid] = node;
        if ((int)level_dfnids.size() <= level)
            level_dfnids.push_back({});
        level_dfnids[level].push_back(dfnid);
        node_level[node] = level;
        for (int son : g[node])
        {
            dfnid += 1;
            dfs(son, dfnid, level + 1);
        }
    }
};