class dsu {
 public:
  vector<int> fa;
  vector<ll> dist;
  int n;

  dsu(int _n) : n(_n) {
    fa.resize(n);
    dist.assign(n, 0);
    iota(fa.begin(), fa.end(), 0);
  }

  int find(int x) {
    if (fa[x] == x) return x;
    int par = fa[x];
    fa[x] = find(fa[x]);
    dist[x] += dist[par];
    return fa[x];
  }

  void unite(int x, int y, ll v) {
    int px = find(x);
    int py = find(y);
    fa[py] = px;
    dist[py] = dist[x] - dist[y] - v;
  }
};