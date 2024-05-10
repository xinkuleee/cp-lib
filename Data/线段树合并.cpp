struct node {
	int sz, sum;
	node *l, *r;
	node() : sz(0), sum(0), l(nullptr), r(nullptr) {}
} pool[N * 20], *cur = pool;

node *newnode() {
	return cur++;
}

void upd(node *rt) {
	if (not rt) return;
	rt->sum = rt->sz > 0;
	if (rt->l) rt->sum += rt->l->sum;
	if (rt->r) rt->sum += rt->r->sum;
}

node *modify(node *rt, int l, int r, int pos, int d) {
	if (not rt) rt = newnode();
	if (l == r) {
		rt->sz += d;
		upd(rt);
		return rt;
	} else {
		int md = (l + r) >> 1;
		if (pos <= md)
			rt->l = modify(rt->l, l, md, pos, d);
		else
			rt->r = modify(rt->r, md + 1, r, pos, d);
		upd(rt);
		return rt;
	}
}

node *merge(node *u, node *v, int l, int r) {
	if (not u) return v;
	if (not v) return u;
	if (l == r) {
		u->sz += v->sz;
		upd(u);
		return u;
	} else {
		int md = (l + r) >> 1;
		u->l = merge(u->l, v->l, l, md);
		u->r = merge(u->r, v->r, md + 1, r);
		upd(u);
		return u;
	}
}

ll query(node *rt, int l, int r) {
	if (not rt) return 0;
	return rt->sum;
}

pair<node *, node *> split(node *rt, int l, int r, int ql, int qr) {
	if (not rt) return {nullptr, nullptr};
	if (ql == l && qr == r) {
		return {nullptr, rt};
	} else {
		int md = (l + r) >> 1;
		if (qr <= md) {
			auto [p1, p2] = split(rt->l, l, md, ql, qr);
			rt->l = p1;
			upd(rt);
			if (not p2) return {rt, nullptr};
			node *u = newnode();
			u->l = p2;
			upd(u);
			return {rt, u};
		} else if (ql > md) {
			auto [p1, p2] = split(rt->r, md + 1, r, ql, qr);
			rt->r = p1;
			upd(rt);
			if (not p2) return {rt, nullptr};
			node *u = newnode();
			u->r = p2;
			upd(u);
			return {rt, u};
		} else {
			auto [p1, p2] = split(rt->l, l, md, ql, md);
			auto [p3, p4] = split(rt->r, md + 1, r, md + 1, qr);
			rt->l = p1, rt->r = p3;
			upd(rt);
			if (not p2 and not p4) return {rt, nullptr};
			node *u = newnode();
			u->l = p2, u->r = p4;
			upd(u);
			return {rt, u};
		}
	}
}