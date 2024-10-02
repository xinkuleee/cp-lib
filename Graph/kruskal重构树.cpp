/**
 * 构建后是一颗二叉树，如果按最小生成树建立的话是大根堆。
 * 性质：原图中两个点间所有路径上的边最大权值的最小值=最小生成树上两点简单路径的边最大权值
 * =kruskal重构树上两点LCA的权值。
 * 重构树中代表原树中的点的节点全是叶子节点，其余节点都代表了一条边的边权。
 * 利用这个性质可以找到点P的简单路径上边权最大值小于lim深度最小的节点。
 * 要求最小权值最大值，可以建最大生成树的重构树从而达到一样的效果。
 */

vector<tuple<ll, ll, ll>> E;
rep(i, 1, m) {
    int u, v, w;
    cin >> u >> v >> w;
    E.emplace_back(w, u, v);
}
ranges::sort(E);
for (auto [w, u, v] : E) {
    u = find(u), v = find(v);
    if (u == v) continue;
    int p = ++idx;
    lim[p] = w;
    fa[u] = p, fa[v] = p;
    e[p].push_back(u);
    e[u].push_back(p);
    e[p].push_back(v);
    e[v].push_back(p);
}