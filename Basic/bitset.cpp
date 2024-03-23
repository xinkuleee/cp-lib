template <int len = 1>
void solve(int n) {
    if (n > len) {
        solve<std::min(len*2, MAXLEN)>(n);
        return;
    }
    // solution using bitset<len>
}

struct Bitset {
    vector<ull> b;
    int n;
    Bitset(int x = 0) {
        n = x;
        b.resize((n + 63) / 64, 0);
    }

    int get(int x) {
        return (b[x >> 6] >> (x & 63)) & 1;
    }

    void set(int x, int y) {
        b[x >> 6] |= 1ULL << (x & 63);
        if (!y) b[x >> 6] ^= 1ULL << (x & 63);
    }

    Bitset &operator&=(const Bitset &another) {
        rep(i, 0, min(SZ(b), SZ(another.b)) - 1) {
            b[i] &= another.b[i];
        }
        return (*this);
    }

    Bitset operator&(const Bitset &another)const {
        return (Bitset(*this) &= another);
    }

    Bitset &operator|=(const Bitset &another) {
        rep(i, 0, min(SZ(b), SZ(another.b)) - 1) {
            b[i] |= another.b[i];
        }
        return (*this);
    }

    Bitset operator|(const Bitset &another)const {
        return (Bitset(*this) |= another);
    }

    Bitset &operator^=(const Bitset &another) {
        rep(i, 0, min(SZ(b), SZ(another.b)) - 1) {
            b[i] ^= another.b[i];
        }
        return (*this);
    }

    Bitset operator^(const Bitset &another)const {
        return (Bitset(*this) ^= another);
    }

    Bitset &operator>>=(int x) {
        if (x & 63) {
            rep(i, 0, SZ(b) - 2) {
                b[i] >>= (x & 63);
                b[i] ^= (b[i + 1] << (64 - (x & 63)));
            }
            b.back() >>= (x & 63);
        }

        x >>= 6;
        rep(i, 0, SZ(b) - 1) {
            if (i + x < SZ(b)) b[i] = b[i + x];
            else b[i] = 0;
        }
        return (*this);
    }

    Bitset operator>>(int x)const {
        return (Bitset(*this) >>= x);
    }

    Bitset &operator<<=(int x) {
        if (x & 63) {
            for (int i = SZ(b) - 1; i >= 1; i--) {
                b[i] <<= (x & 63);
                b[i] ^= b[i - 1] >> (64 - (x & 63));
            }
            b[0] <<= x & 63;
        }

        x >>= 6;
        for (int i = SZ(b) - 1; i >= 0; i--) {
            if (i - x >= 0) b[i] = b[i - x];
            else b[i] = 0;
        }
        return (*this);
    }

    Bitset operator<<(int x)const {
        return (Bitset(*this) <<= x);
    }
};