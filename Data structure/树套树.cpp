struct node {
    int v;
    int l,r,rt;
    node(): v(0),l(0),r(0),rt(0) {}
} seg[maxn*20];
int tot;

ll y_query(int u,int l,int r,int ql,int qr) {
    if (!u) return 0;
    if (l==ql&&r==qr) {
        return seg[u].v;
    }
    int mid=(l+r)>>1;
    if (qr<=mid) return y_query(seg[u].l,l,mid,ql,qr);
    else if (ql>mid) return y_query(seg[u].r,mid+1,r,ql,qr);
    else return y_quer y(seg[u].l,l,mid,ql,mid)
        +y_query(seg[u].r,mid+1,r,mid+1,qr);
}

ll x_query(int u,int l,int r,int xl,int xr,int yl,int yr) {
    if (!u) return 0;
    if (xl==l&&xr==r) {
        return y_query(seg[u].rt,1,n,yl,yr);
    }
    int mid=(l+r)>>1;
    if (xr<=mid) return x_query(seg[u].l,l,mid,xl,xr,yl,yr);
    else if (xl>mid) return x_query(seg[u].r,mid+1,r,xl,xr,yl,yr);
    else return x_query(seg[u].l,l,mid,xl,mid,yl,yr)
        +x_query(seg[u].r,mid+1,r,mid+1,xr,yl,yr);
}

int y_modify(int u,int l,int r,int y,ll v) {
    if (!u) u=++tot;
    if (l==r) {
        seg[u].v+=v;
        return u;
    } else {
        int mid=(l+r)>>1;
        if (y<=mid) seg[u].l=y_modify(seg[u].l,l,mid,y,v);
        else seg[u].r=y_modify(seg[u].r,mid+1,r,y,v);
        seg[u].v=seg[seg[u].l].v+seg[seg[u].r].v;
        return u;
    }
}

int x_modify(int u,int l,int r,int x,int y,ll v) {
    if (!u) u=++tot;
    seg[u].rt=y_modify(seg[u].rt,1,n,y,v);
    if (l==r) return u;
    int mid=(l+r)>>1;
    if (x<=mid) seg[u].l=x_modify(seg[u].l,l,mid,x,y,v);
    else seg[u].r=x_modify(seg[u].r,mid+1,r,x,y,v);
    return u;
}