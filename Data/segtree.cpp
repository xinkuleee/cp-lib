struct info {
	ll sum;
	int sz;
	friend info operator+(const info &a, const info &b) {
		return {(a.sum + b.sum) % mod, a.sz + b.sz};
	}
};

struct tag {
	ll add, mul;
	friend tag operator+(const tag &a, const tag &b) {
		tag res = {(a.add * b.mul + b.add) % mod, a.mul * b.mul % mod};
		return res;
	}
};
info operator+(const info &a, const tag &b) {
	return {(a.sum * b.mul + a.sz * b.add) % mod, a.sz};
}

struct node {
	info val;
	tag t;
} seg[maxn << 2];

void update(int id) {
	seg[id].val = seg[id * 2].val + seg[id * 2 + 1].val;
}
void settag(int id, tag t) {
	seg[id].val = seg[id].val + t;
	seg[id].t = seg[id].t + t;
}
void pushdown(int id) {
	if (seg[id].t.mul == 1 and seg[id].t.add == 0) return;
	settag(id * 2, seg[id].t);
	settag(id * 2 + 1, seg[id].t);
	seg[id].t.mul = 1;
	seg[id].t.add = 0;
}
void build(int l, int r, int id) {
	seg[id].t = {0, 1};
	if (l == r) {
		seg[id].val = {a[l], 1};
	} else {
		int mid = (l + r) >> 1;
		build(l, mid, id * 2);
		build(mid + 1, r, id * 2 + 1);
		update(id);
	}
}
void change(int l, int r, int id, int ql, int qr, tag t) {
	if (l == ql && r == qr) {
		settag(id, t);
	} else {
		int mid = (l + r) >> 1;
		pushdown(id);
		if (qr <= mid) {
			change(l, mid, id * 2, ql, qr, t);
		} else if (ql > mid) {
			change(mid + 1, r, id * 2 + 1, ql, qr, t);
		} else {
			change(l, mid, id * 2, ql, mid, t);
			change(mid + 1, r, id * 2 + 1, mid + 1, qr, t);
		}
		update(id);
	}
}
info query(int l, int r, int id, int ql, int qr) {
	if (l == ql && r == qr) {
		return seg[id].val;
	} else {
		int mid = (l + r) >> 1;
		pushdown(id);
		if (qr <= mid)
			return query(l, mid, id * 2, ql, qr);
		else if (ql > mid)
			return query(mid + 1, r, id * 2 + 1, ql, qr);
		else
			return query(l, mid, id * 2, ql, mid) +
				   query(mid + 1, r, id * 2 + 1, mid + 1, qr);
	}
}
ll search(int l, int r, int id, int ql, int qr, int d) {
	if (ql == l && qr == r) {
		int mid = (l + r) / 2;
		// if (l != r) pushdown(id); ...
		if (seg[id].val < d)
			return -1;
		else {
			if (l == r)
				return l;
			else if (seg[id * 2].val >= d)
				return search(l, mid, id * 2, ql, mid, d);
			else
				return search(mid + 1, r, id * 2 + 1, mid + 1, qr, d);
		}
	} else {
		int mid = (l + r) >> 1;
		// pushdown(id); ...
		if (qr <= mid)
			return search(l, mid, id * 2, ql, qr, d);
		else if (ql > mid)
			return search(mid + 1, r, id * 2 + 1, ql, qr, d);
		else {
			int tmp = search(l, mid, id * 2, ql, mid, d);
			if (tmp != -1)
				return tmp;
			else
				return search(mid + 1, r, id * 2 + 1, mid + 1, qr, d);
		}
	}
}
