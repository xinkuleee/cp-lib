vector<PII> e[N];

template <typename T>
void add(int u, int v, T w) {
	e[u].eb(v, w);
}

template <typename T>
T prim(vector<pair<int, T>> *g, int start) {
	const T inf = numeric_limits<T>::max() / 4;
	T res = 0;
	vector<T> dist(n, inf);
	dist[start] = 0;
	priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> s;
	s.emplace(dist[start], start);
	vector<int> was(n, 0);
	while (!s.empty()) {
		T expected = s.top().first;
		int i = s.top().second;
		s.pop();
		if (dist[i] != expected || was[i]) {
			continue;
		}
		was[i] = 1;
		res += expected;
		for (auto [to, cost] : g[i]) {
			if (cost < dist[to]) {
				dist[to] = cost;
				s.emplace(dist[to], to);
			}
		}
	}
	return res;
}