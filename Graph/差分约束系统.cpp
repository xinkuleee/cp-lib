/**
	Description:
	求解方程组 x_u - x_v <= w_i, 求出的x_i为满足条件的最大值
 	转化为x_u <= x_v + w_i
	问题等价于求最短路（bellmanford或Floyd）
	即加一条有向边add(u, v, w), dist[v] = min(dist[v], dist[u] + w)
	求最小值（满足条件情况下尽量小）等价于求(-x_i)最大（或者转化为求最长路)
 	求非负解只需要添加超级节点S，S向各个点连边（S + 0 <= x_i），再设dist[S] = 0
 */
void solve() {
	cin >> n >> m;
	vector<int> dist(n, 0);
	vector<vector<PII>> g(n);
	rep(i, 0, m - 1) {
		int u, v, w;
		cin >> u >> v >> w;
		u--, v--;
		g[u].eb(v, -w);
	}
	bool ok = 1;
	rep(i, 1, n) {
		bool upd = 0;
		rep(u, 0, n - 1) {
			for (auto [v, w] : g[u]) {
				if (dist[v] < dist[u] + w) {
					dist[v] = dist[u] + w;
					upd = 1;
				}
			}
		}
		if (!upd) break;
		// 仍然有约束未满足
		if (i == n && upd) ok = 0;
	}
	if (!ok) {
		return cout << -1 << '\n', void();
	}
	rep(i, 0, n - 1) {
		cout << dist[i] << " \n"[i == n - 1];
	}
}