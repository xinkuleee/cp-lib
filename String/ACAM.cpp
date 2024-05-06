const int M = 2, N = 2.1e5;
struct node {
	node *son[M], *go[M], *fail;
	int cnt, vis, ins;
} pool[N], *cur = pool, *q[N], *root;

node *newnode() { return cur++; }
int t, n;

void build() {
	t = 0;
	q[t++] = root;
	for (int i = 0; i < t; i++) {
		node *u = q[i];
		for (int j = 0; j < M; j++) {
			if (u->son[j]) {
				u->go[j] = u->son[j];
				if (u != root)
					u->go[j]->fail = u->fail->go[j];
				else
					u->go[j]->fail = root;
				q[t++] = u->son[j];
			} else {
				if (u != root)
					u->go[j] = u->fail->go[j];
				else
					u->go[j] = root;
			}
		}
	}
}

void insert(string &s) {
	node *cur = root;
	for (auto c : s) {
		int w = c - '0';
		if (!cur->son[w]) {
			cur->son[w] = newnode();
		}
		cur = cur->son[w];
	}
	cur->cnt = 1;
}