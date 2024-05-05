namespace compact {
const int LOGN=18;
int l[N],r[N],tot,p[N][20],n;
map<int,int> cv;
int lca(int u,int v) {
    if (dep[u]>dep[v]) swap(u,v);
    per(i,LOGN-1,0) if (dep[p[v][i]]>=dep[u]) v=p[v][i];
    if (u==v) return u;
    per(i,LOGN-1,0) if (p[v][i]!=p[u][i]) u=p[u][i],v=p[v][i];
    return p[u][0];
}
void dfs(int u,int f) {
    l[u]=++tot; dep[u]=dep[f]+1; p[u][0]=f;
    vec[dep[u]].pb(u);
    for (auto v:vE[u]) {
        if (v==f) continue;
        dfs(v,u);
    }
    r[u]=tot;
}
void build(int _n) {
    n=_n; tot=0;
    dfs(1,0);
    rep(j,1,LOGN-1) rep(i,1,n) p[i][j]=p[p[i][j-1]][j-1];
}
 
bool cmp(int u,int v) { return l[u]<l[v]; }
vector<PII> compact(VI v) {
    int m=SZ(v);
    vector<PII> E;
    sort(all(v),cmp);
    rep(i,0,m-2) {
        int w=lca(v[i],v[i+1]);
        v.pb(w);
    }
    v.pb(0);
    v.pb(1);
    sort(all(v),cmp); 
    v.erase(unique(all(v)),v.end());
    cv.clear();
    per(i,SZ(v)-1,1) {
        int u=v[i];
        while (1) {
            auto it=cv.lower_bound(l[u]);
            if (it==cv.end()||it->fi>r[u]) break;
            E.pb(mp(u,v[it->se]));
            cv.erase(it);
        }
        cv[l[u]]=i;
    }
    return E;
}
};