#include <bits/stdc++.h>
using namespace std;
using ll = long long;

// L <= R, 左边完全匹配
// 最小权完备匹配

// 带权匹配：使得该二分图的权值和最大（或最小）的匹配。
// 最大匹配：使得该二分图边数最多的匹配。
// 完备匹配：使得点数较小的点集中每个点都被匹配的匹配。
// 完美匹配：所有点都被匹配的匹配。
// 定理1：最大匹配数 = 最小点覆盖数（Konig 定理）
// 定理2：最大匹配数 = 最大独立数
// 定理3：最小路径覆盖数 = 顶点数 - 最大匹配数

// 二分图的最小点覆盖
// 定义：在二分图中，求最少的点集，使得每一条边至少都有端点在这个点集中。
// 二分图的最小点覆盖 = 二分图的最大匹配

// 二分图的最少边覆盖
// 定义：在二分图中，求最少的边，使得他们覆盖所有的点，并且每一个点只被一条边覆盖。
// 二分图的最少边覆盖 = 点数 - 二分图的最大匹配

// 二分图的最大独立集
// 定义：在二分图中，选最多的点，使得任意两个点之间没有直接边连接。
// 二分图的最大独立集 = 点数 - 二分图的最大匹配

template<class T>
pair<T, vector<int>> hungarian(const vector<vector<T>> &a) {
	if (a.empty()) return {0, {}};
	int n = a.size() + 1, m = a[0].size() + 1;
	vector<T> u(n), v(m); // 顶标
	vector<int> p(m), ans(n - 1);
	for (int i = 1; i < n; i++) {
		p[0] = i;
		int j0 = 0;
		vector<T> dist(m, numeric_limits<T>::max());
		vector<int> pre(m, -1);
		vector<bool> done(m + 1);
		do { // dijkstra
			done[j0] = true;
			int i0 = p[j0], j1;
			T delta = numeric_limits<T>::max();
			for (int j = 1; j < m; j++) if (!done[j]) {
				auto cur = a[i0 - 1][j - 1] - u[i0] - v[j];
				if (cur < dist[j]) dist[j] = cur, pre[j] = j0;
				if (dist[j] < delta) delta = dist[j], j1 = j;
			}
			for (int j = 0; j < m; j++) {
				if (done[j]) u[p[j]] += delta, v[j] -= delta;
				else dist[j] -= delta;
			}
			j0 = j1;
		} while (p[j0]);
		while (j0) { // update alternating path
			int j1 = pre[j0];
			p[j0] = p[j1], j0 = j1;
		}
	}
	for (int j = 1; j < m; j++) {
		if (p[j]) ans[p[j] - 1] = j - 1;
	}
	return {-v[0], ans}; // min cost
}

int L, R, m;
int main() {
	scanf("%d%d%d", &L, &R, &m);
	R = max(L, R);
	auto a = vector<vector<ll>>(L, vector<ll>(R, 0));
	for (int i = 0; i < m; i++) {
		int u, v, w;
		scanf("%d%d%d", &u, &v, &w);
		--u; --v;
		a[u][v] = -w;
	}
	auto [val, ans] = hungarian(a);
	printf("%lld\n", -val);
	for (int i = 0; i < L; i++) {
		if (a[i][ans[i]] >= 0) ans[i] = -1;
		printf("%d%c", ans[i] + 1, " \n"[i == L - 1]);
	}
}