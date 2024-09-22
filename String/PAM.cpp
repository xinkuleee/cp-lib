struct PAM {
	struct T {
		array<int, 10> tr;
		int fail, len, tag;
		T() : fail(0), len(0), tag(0) {
			tr.fill(0);
		}
	};
	vector<T> t;
	vector<int> stk;
	int newnode(int len) {
		t.emplace_back();
		t.back().len = len;
		return (int)t.size() - 1;
	}
	PAM() : t(2) {
		t[0].fail = 1, t[0].len = 0;
		t[1].fail = 0, t[1].len = -1;
		stk.push_back(-1);
	}
	int getfail(int v) {
		while (stk.end()[-2 - t[v].len] != stk.back()) {
			v = t[v].fail;
		}
		return v;
	}
	int insert(int lst, int c, int td) {
		stk.emplace_back(c);
		int x = getfail(lst);
		if (!t[x].tr[c]) {
			int u = newnode(t[x].len + 2);
			t[u].fail = t[getfail(t[x].fail)].tr[c];
			t[x].tr[c] = u;
		}
		t[t[x].tr[c]].tag += td;
		return t[x].tr[c];
	}
	int build(int n) {
		int ans = 0;
		for (int i = (int)t.size() - 1; i > 1; i--) {
			t[t[i].fail].tag += t[i].tag;
			if (t[i].len > n) {
				continue;
			}
			ans = (ans + 1ll * t[i].tag * t[i].tag % M * t[i].len) % M;
		}
		return ans;
	}
};