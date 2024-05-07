int n, m, a[N];
vector<int> e[N];
int l[N], r[N], idx[N];
int sz[N], hs[N], tot, top[N], dep[N], fa[N];

struct info {
	int maxv, sum;
};

info operator + (const info &l, const info &r) {
	return (info){max(l.maxv, r.maxv), l.sum + r.sum};
}

struct node {
	info val;
} seg[N * 4];

// [l, r]

void update(int id) {
	seg[id].val = seg[id * 2].val + seg[id * 2 + 1].val;
}

void build(int id, int l, int r) {
	if (l == r) {
		// l号点， DFS序中第l个点‼️
		seg[id].val = {a[idx[l]], a[idx[l]]};
	} else {
		int mid = (l + r) / 2;
		build(id * 2, l, mid);
		build(id * 2 + 1, mid + 1, r);
		update(id);
	}
}

void change(int id, int l, int r, int pos, int val) {
	if (l == r) {
		seg[id].val = {val, val};
	} else {
		int mid = (l + r) / 2;
		if (pos <= mid) change(id * 2, l, mid, pos, val);
		else change(id * 2 + 1, mid + 1, r, pos, val);
		update(id);
	}
} 

info query(int id, int l, int r, int ql, int qr) {
	if (l == ql && r == qr) return seg[id].val;
	int mid = (l + r) / 2;
	if (qr <= mid) return query(id * 2, l, mid, ql, qr);
	else if (ql > mid) return query(id * 2 + 1, mid + 1, r, ql,qr);
	else {
		return query(id * 2, l, mid, ql, mid) + 
			query(id * 2 + 1, mid + 1, r, mid + 1, qr);
	}
}

// 第一遍DFS，子树大小，重儿子，父亲，深度
void dfs1(int u,int f) {
	sz[u] = 1;
	hs[u] = -1;
	fa[u] = f;
	dep[u] = dep[f] + 1;
	for (auto v : e[u]) {
		if (v == f) continue;
		dfs1(v, u);
		sz[u] += sz[v];
		if (hs[u] == -1 || sz[v] > sz[hs[u]])
			hs[u] = v;
	}
}

// 第二遍DFS，每个点DFS序，重链上的链头的元素。
void dfs2(int u, int t) {
	top[u] = t;
	l[u] = ++tot;
	idx[tot] = u;
	if (hs[u] != -1) {
		dfs2(hs[u], t);
	}
	for (auto v : e[u]) {
		if (v != fa[u] && v != hs[u]) {
			dfs2(v, v);
		}
	}
	r[u] = tot;
}

int LCA(int u, int v) {
	while (top[u] != top[v]) {
		if (dep[top[u]] < dep[top[v]]) v = fa[top[v]];
		else u = fa[top[u]];
	}
	if (dep[u] < dep[v]) return u;
	else return v;
}

info query(int u,int v) {
	info ans{(int)-1e9, 0};
	while (top[u] != top[v]) {
		if (dep[top[u]] < dep[top[v]]) {
			ans = ans + query(1, 1, n, l[top[v]], l[v]);
			v = fa[top[v]];
		} else {
			ans = ans + query(1, 1, n, l[top[u]], l[u]);
			u = fa[top[u]];
		}
	}
	if (dep[u] <= dep[v]) ans = ans + query(1, 1, n, l[u], l[v]);
	else ans = ans + query(1, 1, n, l[v], l[u]);
	return ans;
}