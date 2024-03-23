struct node {
    int son[26], go[26];
    int fail, end, cnt;
    node() {
        rep(i, 0, 25) { son[i] = 0; go[i] = 0; }
        fail = 0; end = 0; cnt = 0;
    }
} ac[maxn];
int tot, d[maxn];

void insert(string s, int id) {
    int u = 0;
    for (int i = 0; i < s.size(); i++) {
        int w = s[i] - 'a';
        if (!ac[u].son[w])
            ac[u].son[w] = ++tot;
        u = ac[u].son[w];
    }
    // ac[u].end++;
    d[id] = u;
}

void get_fail() {
    queue<int> q;
    for (int i = 0; i < 26; i++) {
        if (ac[0].son[i]) {
            ac[0].go[i] = ac[0].son[i];
            ac[ac[0].son[i]].fail = 0;
            q.push(ac[0].son[i]);
        } else {
            ac[0].go[i] = 0;
        }
    }
    while (q.size()) {
        int u = q.front(); q.pop();
        ans.pb(u);
        for (int i = 0; i < 26; i++) {
            if (ac[u].son[i]) {
                ac[u].go[i] = ac[u].son[i];
                ac[ac[u].son[i]].fail = ac[ac[u].fail].go[i];
                q.push(ac[u].son[i]);
            } else {
                ac[u].go[i] = ac[ac[u].fail].go[i];
            }
        }
    }
}

void query(string s) {
    int u = 0;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == ' ') u = 0;
        else u = ac[u].go[s[i] - 'a'];
        ac[u].cnt++;
    }
}

void solve() {
    cin >> n;
    string s, t;
    rep(i, 1, n) {
        cin >> t;
        insert(t, i);
        s += t;
        if (i != n) s.pb(' ');
    }
    get_fail();
    query(s);
    reverse(all(ans));
    for (auto y : ans) {
        ac[ac[y].fail].cnt += ac[y].cnt;
    }
    rep(i, 1, n) cout << ac[d[i]].cnt << '\n';
}
