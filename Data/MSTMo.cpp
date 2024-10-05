#include <bits/stdc++.h>
using namespace std;
#define rep(i,a,n) for (int i=a;i<n;i++)
#define per(i,a,n) for (int i=n-1;i>=a;i--)
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
typedef vector<int> VI;
typedef long long ll;
typedef pair<int,int> PII;
typedef double db;
mt19937 mrand(random_device{}()); 
const ll mod=1000000007;
int rnd(int x) { return mrand() % x;}
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a,ll b) { return b?gcd(b,a%b):a;}
// head

const int N=1010000;
int a[N];
namespace Mo {
  int Q,l[N],r[N],f[N],l0,r0,ans[N],n;
  VI ne[N];
  struct point {
    int x, y, o;
    point(int a, int b, int c): x(a), y(b), o(c) {}
  };
  inline bool operator<(const point &a, const point &b) {
    if (a.x != b.x) return a.x > b.x;
    else return a.y < b.y;
  }
  vector<point> p;
  struct edge {
    int s, t, d;
    edge(const point &a, const point &b): s(a.o), t(b.o),
      d(abs(a.x - b.x) + abs(a.y - b.y)) {}
  };
  inline bool operator<(const edge &a, const edge &b) {return a.d < b.d;}
  vector<edge> e;
  int g[N],z[N];
  int cc,cnt[101000];
  void addedge() {
    sort(all(p));
      memset(g,0,sizeof(g));
      z[0]=N;
    rep(i,0,SZ(p)) z[i+1]=p[i].x-p[i].y;
    rep(i,0,SZ(p)) {
          int k = 0, t = p[i].x + p[i].y;
          for (int j = t; j; j -= j & -j)
              if (z[g[j]] < z[k]) k = g[j];
          if (k) e.pb(edge(p[i], p[k - 1]));
          k = z[i + 1];
          for (int j = t; j <N; j += j & -j)
              if (k < z[g[j]]) g[j] = i + 1;
      }
  }
  void updata(int i, bool j,bool k=0) {
    // j=1 insert  j=0 delete
    // k=0 left k=1 right
    if (j==1) {
      cnt[a[i]]++;
      if (cnt[a[i]]%2==0) cc++;
    } else {
      if (cnt[a[i]]%2==0) cc--;
      cnt[a[i]]--;
    }
  }
  void init(int l,int r) {
    for (int i=l;i<=r;i++) {
      cnt[a[i]]++;
      if (cnt[a[i]]%2==0) cc++; 
    }
  }
  inline int query() {
    return cc;
  }
  int find(int x) { if (f[x] != x) f[x] = find(f[x]); return f[x];}
  void dfs(int i,int p) {
    int l1 = l[i], r1 = r[i];
    per(j,l1,l0) updata(j,1,0);
    rep(j,r0+1,r1+1) updata(j,1,1);
    rep(j,l0,l1) updata(j,0,0);
    per(j,r1+1,r0+1) updata(j,0,1);
    ans[i]=query();l0=l1;r0=r1;
    rep(j,0,SZ(ne[i])) if (ne[i][j]!=p) dfs(ne[i][j],i);
  }
  void solve() {
    p.clear();e.clear();
    rep(i,1,Q+1) ans[i]=0;
    rep(i,1,Q+1) p.pb(point(l[i],r[i],i));
    addedge();
    rep(i,0,SZ(p)) p[i].y =n-p[i].y+1;
    addedge();
    rep(i,0,SZ(p)) {
      int j =n-p[i].x+1;
      p[i].x = p[i].y; p[i].y = j;
    }
    addedge();
    rep(i,0,SZ(p)) p[i].x=n-p[i].x+1;
    addedge();
    sort(all(e));
    rep(i,1,Q+1) ne[i].clear(),f[i]=i;
    rep(i,0,SZ(e)) {
      int j=e[i].s,k=e[i].t;
      if (find(j)!=find(k)) f[f[j]]=f[k],ne[j].pb(k),ne[k].pb(j);
    }
    l0=l[1];r0=r[1];
    init(l0,r0);
    dfs(1,0);
  }
}

int main() {
  scanf("%d",&Mo::n);
  for (int i=1;i<=Mo::n;i++) scanf("%d",a+i);
  scanf("%d",&Mo::Q);
  rep(i,1,Mo::Q+1) scanf("%d%d",&Mo::l[i],&Mo::r[i]);
  Mo::solve();
  rep(i,1,Mo::Q+1) printf("%d\n",Mo::ans[i]);
}
