const int SZ = 1.1e5;
template <class T>
struct node {
	T val = 0;
	node<T>* c[2];
	node() { c[0] = c[1] = NULL; }
	void upd(int ind, T v, int L = 0, int R = SZ - 1) {  // add v
		if (L == ind && R == ind) {
			val += v;
			return;
		}
		int M = (L + R) / 2;
		if (ind <= M) {
			if (!c[0]) c[0] = new node();
			c[0]->upd(ind, v, L, M);
		} else {
			if (!c[1]) c[1] = new node();
			c[1]->upd(ind, v, M + 1, R);
		}
		val = 0;
		rep(i, 0, 1) if (c[i]) val += c[i]->val;
	}
	T query(int lo, int hi, int L = 0, int R = SZ - 1) {  // query sum of segment
		if (hi < L || R < lo) return 0;
		if (lo <= L && R <= hi) return val;
		int M = (L + R) / 2;
		T res = 0;
		if (c[0]) res += c[0]->query(lo, hi, L, M);
		if (c[1]) res += c[1]->query(lo, hi, M + 1, R);
		return res;
	}
	void UPD(int ind, node* c0, node* c1, int L = 0, int R = SZ - 1) {  // for 2D segtree
		if (L != R) {
			int M = (L + R) / 2;
			if (ind <= M) {
				if (!c[0]) c[0] = new node();
				c[0]->UPD(ind, c0 ? c0->c[0] : NULL, c1 ? c1->c[0] : NULL, L, M);
			} else {
				if (!c[1]) c[1] = new node();
				c[1]->UPD(ind, c0 ? c0->c[1] : NULL, c1 ? c1->c[1] : NULL, M + 1, R);
			}
		}
		val = (c0 ? c0->val : 0) + (c1 ? c1->val : 0);
	}
};

/**
 * Description: BIT of SegTrees. $x\in (0,SZ), y\in [0,SZ)$.
 * Memory: O(N\log^2 N)
 * Source: USACO Mowing the Field
 * Verification:
 * USACO Mowing the Field
 * http://www.usaco.org/index.php?page=viewproblem2&cpid=722 (13/15, 15/15 and 1857ms with BumpAllocator)
 */

#include "../1D Range Queries (9.2)/SparseSeg (9.2).h"

template <class T>
struct BITseg {
	node<T> seg[SZ];
	BITseg() { fill(seg, seg + SZ, node<T>()); }
	void upd(int x, int y, int v) {  // add v
		for (; x < SZ; x += x & -x) seg[x].upd(y, v);
	}
	T query(int x, int yl, int yr) {
		T res = 0;
		for (; x; x -= x & -x) res += seg[x].query(yl, yr);
		return res;
	}
	T query(int xl, int xr, int yl, int yr) {  // query sum of rectangle
		return query(xr, yl, yr) - query(xl - 1, yl, yr);
	}
};

/**
 * Description: SegTree of SegTrees. $x,y\in [0,SZ).$
 * Memory: O(N\log^2 N)
 * Source: USACO Mowing the Field
 * Verification:
 * http://www.usaco.org/index.php?page=viewproblem2&cpid=722 (9/15 w/ BumpAllocator)
 * http://www.usaco.org/index.php?page=viewproblem2&cpid=601 (4238 ms, 2907 ms w/ BumpAllocator)
 */

#include "../1D Range Queries (9.2)/SparseSeg (9.2).h"

template <class T>
struct Node {
	node<T> seg;
	Node* c[2];
	Node() { c[0] = c[1] = NULL; }
	void upd(int x, int y, T v, int L = 0, int R = SZ - 1) {  // add v
		if (L == x && R == x) {
			seg.upd(y, v);
			return;
		}
		int M = (L + R) / 2;
		if (x <= M) {
			if (!c[0]) c[0] = new Node();
			c[0]->upd(x, y, v, L, M);
		} else {
			if (!c[1]) c[1] = new Node();
			c[1]->upd(x, y, v, M + 1, R);
		}
		seg.upd(y, v);  // only for addition
		// seg.UPD(y,c[0]?&c[0]->seg:NULL,c[1]?&c[1]->seg:NULL);
	}
	T query(int x1, int x2, int y1, int y2, int L = 0, int R = SZ - 1) {  // query sum of rectangle
		if (x1 <= L && R <= x2) return seg.query(y1, y2);
		if (x2 < L || R < x1) return 0;
		int M = (L + R) / 2;
		T res = 0;
		if (c[0]) res += c[0]->query(x1, x2, y1, y2, L, M);
		if (c[1]) res += c[1]->query(x1, x2, y1, y2, M + 1, R);
		return res;
	}
};