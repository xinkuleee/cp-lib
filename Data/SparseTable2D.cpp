// lg[1] = 0;
// rep(i, 2, N - 1) {
//     lg[i] = lg[i / 2] + 1;
// }
// int k = log2(r - l + 1); very slow!!!
// int k = __lg(r - l + 1);
// int k = lg[r - l + 1];
// int k = 32 - __builtin_clz(r - l + 1) - 1;
vector<vector<int>> sparse[12];

int query(int x, int y, int d) {
  int k = __lg(d);
  int s = d - bit(k);
  return min({sparse[k][x][y], sparse[k][x + s][y], sparse[k][x][y + s], sparse[k][x + s][y + s]});
}

void build() {
  rep(i, 1, n) rep(j, 1, m) sparse[0][i][j] = mat[i][j];
  rep(k, 1, 11) rep(i, 1, n) rep(j, 1, m) {
    int d = bit(k - 1);
    if (i + d > n || j + d > m) continue;
    sparse[k][i][j] = min({sparse[k - 1][i][j], sparse[k - 1][i + d][j], sparse[k - 1][i][j + d], sparse[k - 1][i + d][j + d]});
  }
}