struct Hash {
    string s;
    vector<long long> p1, h1;
    vector<long long> p2, h2;
    static const int M1 = 1e9 + 7, w1 = 101, M2 = 998244353, w2 = 91;
    void build(string _s) {
        h1.clear();
        h2.clear();
        s = _s;
        h1.push_back(0);
        h2.push_back(0);
        p1.push_back(1);
        p2.push_back(1);

        for (int i = 0; i < (int)s.size(); i++) {
            h1.push_back((h1.back() * w1 % M1 + s[i] - 'a' + 1) % M1);
            h2.push_back((h2.back() * w2 % M2 + s[i] - 'a' + 1) % M2);
            p1.push_back(p1.back() * w1 % M1);
            p2.push_back(p2.back() * w2 % M2);
        }
    }
    pair<long long, long long> hash(int l, int r) {
        long long res1 = (h1[r] - h1[l - 1] * p1[r - l + 1] % M1) % M1;
        long long res2 = (h2[r] - h2[l - 1] * p2[r - l + 1] % M2) % M2;
        return {(res1 + M1) % M1, (res2 + M2) % M2};
    }
};
