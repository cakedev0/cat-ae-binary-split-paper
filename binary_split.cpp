#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/hash_policy.hpp>
using namespace __gnu_pbds;
#include <chrono>   
using namespace std;
#define cs const
#define re register
#define pb emplace_back
#define pii pair<int,int>
#define ll long long
#define y1 shinkle
#define fi first
#define se second
#define bg begin
template<typename tp>inline void chemx(tp &a,tp b){a=max(a,b);}
template<typename tp>inline void chemn(tp &a,tp b){a=min(a,b);}

cs int N=20000005;
cs double eps=1e-8;
cs double inf=1e18;
struct node{
    double a,b,c;
    node(double _a=0,double _b=0,double _c=0):a(_a),b(_b),c(_c){}
};
struct func{
    /*
    store the piecewise linear function
    pos:coordinate of x
    slope: the slope after the x
    f:f(x)
    */
    int n;
    vector<double>pos;
    vector<double>slope;
    vector<double> f;   
    double querymn(double l,double r){// return the minimal value in [l,r]
        double res=inf;
        for(int i=0;i<pos.size();i++){
            if(l<=pos[i]&&pos[i]<=r)res=min(res,f[i]);
            if(i&&pos[i-1]<l&&l<=pos[i])res=min(res,f[i-1]+slope[i-1]*(l-pos[i-1]));
            if(i&&pos[i-1]<r&&r<=pos[i])res=min(res,f[i-1]+slope[i-1]*(r-pos[i-1]));
        }return res;
    }
    double calc (double x)const{// return value of f(x)
        if(!pos.size())return 0;
        if(x<pos[0])return f[0]-(slope[0]-2)*(pos[0]-x);
        int l=0,r=pos.size()-1,res=0;
        while(l<=r){
            int mid=(l+r)/2;
            if(pos[mid]<=x)res=mid,l=mid+1;
            else r=mid-1;
        }
        return f[res]+slope[res]*(x-pos[res]);
    }
    node query(double x)const{//return f(x),index of x',slope of x';
        if(x<pos[0])return node(f[0]-(slope[0]-2)*(pos[0]-x),0,slope[0]-2);
        int l=0,r=pos.size()-1,res=0;
        while(l<=r){
            int mid=(l+r)/2;
            if(pos[mid]<=x)res=mid,l=mid+1;
            else r=mid-1;
        }
        return node(f[res]+slope[res]*(x-pos[res]),res+1,slope[res]);
    }
    node findf(double x)const{// return f'(a),f(a),the slope of f'(a)
        node tmp=query(x);
        ll n=pos.size();
        ll l=tmp.b-1,r=n-1,res=l;
        if(l<0)l=0,res=0;
        if(l+1<n&&f[l]>tmp.a)l++,res++;
        while(l<=r){
            ll mid=(l+r)/2;
            if(f[mid]<=tmp.a)res=mid,l=mid+1;
            else r=mid-1;
        }
        if(slope[res]==0){
            // for(int i=0;i<pos.size();i++){
            //     cout<<pos[i]<<" "<<slope[i]<<" "<<f[i]<<'\n';
            // }
            // cout<<x<<" "<<tmp.a<<" "<<res<<" "<<f[res]<<" "<<slope[res]<<'\n';
            // assert(0);
            return node(1e18,tmp.a,slope[res]);
        }
        double fa=(pos[res]+(tmp.a-f[res])/slope[res]);
        return node(fa,tmp.a,slope[res]);
    }
};

//build the piecewise linear function on pointset
func build_func(cs vector<double> &a){
    func ff;
    double sm=0,v0=0,slope=a.size();
    for(double x:a)sm+=x,ff.pos.pb(x);
    for(double x:a)v0+=x-a[0];
    ff.f.pb(v0);
    slope=-slope;
    slope+=2;
    ff.slope.pb(slope);
    for(int i=1;i<a.size();i++){
        v0+=(a[i]-a[i-1])*slope;slope+=2;
        assert(v0>=0);
        ff.f.pb(v0);ff.slope.pb(slope);
    }
    ff.n=ff.f.size();
    return ff;
}

//Reduce the point in function to coordinate in [l,r]
void Reduce(func &x,double l,double r){
    if(!x.pos.size())return;
    if(x.pos.back()<=l){
        double ta=x.pos.back(),tb=x.f.back(),tc=x.slope.back();
        x.pos.clear(),x.pos.pb(l);
        x.f.clear(),x.f.pb(tb+tc*(l-ta));
        x.slope.clear(),x.slope.pb(tc);
    }
    int i=-1,n=x.pos.size();
    while(i+1<x.pos.size()&&x.pos[i+1]<=l)i++;
    i--;
    if(i>=0){
        x.pos.erase(x.pos.begin(),x.pos.begin()+i+1);
        x.f.erase(x.f.begin(),x.f.begin()+i+1);
        x.slope.erase(x.slope.begin(),x.slope.begin()+i+1);
    }
    while(x.pos.size()>1&&r<x.pos.back()){
        x.pos.pop_back();
        x.f.pop_back();
        x.slope.pop_back();
    }
    x.n=x.f.size();
}

double v[N];
//coordinate of index i
int tt,n;
double sv[N],sl[N];
gp_hash_table<double, int> id;
//which index of coordinate
vector<func> Fc;
//merge F into A in range[v[l],v[r]]
void merge(func &res,cs func &A,cs vector<int> &F,int l,int r){

    //update value and slope in array to merge functions
    double lst;int pos;
    for(cs int &x:F){
        func *pp=&Fc[x];
        int n=pp->pos.size();
        node tmp=pp->query(v[l]);
        sv[l]+=tmp.a;
        sl[l]+=tmp.c;
        lst=tmp.c;
        for(int i=round(tmp.b);i<pp->pos.size();i++){
            if(pp->pos[i]>v[r])break;
            pos=id[pp->pos[i]];
            sl[pos]+=pp->slope[i]-lst;
            lst=pp->slope[i];
        }
    }
    if(A.pos.size()){
        int n=A.pos.size();
        node tmp=A.query(v[l]);
        sv[l]+=tmp.a;
        sl[l]+=tmp.c;
        lst=tmp.c;
        for(int i=round(tmp.b);i<A.pos.size();i++){
            if(A.pos[i]>v[r])break;
            pos=id[A.pos[i]];
            sl[pos]+=A.slope[i]-lst;
            lst=A.slope[i];
        }
    }
    double sm=0,slope=0;lst=0;
    res.pos.clear(),res.f.clear(),res.slope.clear();
    {
        for(int i=l;i<=r;i++){
            if(sl[i]||sv[i]){
                sm+=slope*(v[i]-lst)+sv[i];
                slope+=sl[i];
                res.pos.emplace_back(v[i]),
                res.f.emplace_back(sm),
                res.slope.emplace_back(slope);
                lst=v[i];

            }
        }
        for(int i=l;i<=r;i++){
            sl[i]=sv[i]=0;
        }
    }
}
pair<double,ll> calc_row_min(int a,int lb,int rb,cs func &EA,cs func &EB,cs vector<int> &F){
    //calc the minimal value of F in (a,[lb,rb])
    //update value and slope 
    double mn=inf;ll mnpos=-1;
    if(lb<a)lb=a;

    for(cs int &x:F){
        func *pp=&Fc[x];
        node ret=pp->findf(v[a]);
        double fa=ret.a;
        if(fa<=v[lb]){
            sv[lb]+=ret.b;continue;
        }
        else{
            //when f(b) <= f(a) choose b
            int pos,n=pp->pos.size(),l,r,res;
            node tmp=pp->query(v[lb]);
            sv[lb]+=tmp.a;
            sl[lb]+=tmp.c;
            for(int i=tmp.b;i<pp->pos.size();i++){
                if(pp->pos[i]>v[rb])break;
                pos=id[pp->pos[i]];
                if(pp->pos[i]>fa){
                    break;
                }
                sl[pos]+=2;
            }
            // when f(a) < f(b),choose a
            if(v[rb]>=fa){
                l=lb,r=rb,res=-1;
                while(l<=r){
                    int mid=(l+r)/2;
                    if(fa<=v[mid])res=mid,r=mid-1;
                    else l=mid+1;
                }
                assert(res!=-1);
                double tmp=pp->query(v[res]).a;
                sv[res]=sv[res]-tmp+ret.b;
                sl[res]=sl[res]-ret.c;
            }
        }
    }
    //calc value in each coordinate
    double sm=0,slope=0;
    double vala=EA.calc(v[a]);
    for(int b=lb;b<=rb;b++){
        double res=vala+EB.calc(v[b]);
        sm+=slope*(v[b]-v[b-1])+sv[b];
        res+=sm,slope+=sl[b];
        if(res<mn)mn=res,mnpos=b;
    }
    for(int i=lb;i<=rb;i++)sl[i]=sv[i]=0;
    return make_pair(mn,mnpos);
}
void write(cs vector<int> &x){
    for(int v:x)cout<<v<<" ";puts("");
}
//divide and conquer in ([la,ra],[lb,rb])
//EA: the sum of function choose a,so as EB
//F:the set of funtions
struct Anstype{
    double ans;
    ll ans_a,ans_b;
    Anstype(double _ans=0,ll _a=0,ll _b=0):ans(_ans),ans_a(_a),ans_b(_b){}
    bool operator<(const Anstype &x)const{
        return ans<x.ans;
    }
};
Anstype divide(int la,int ra,int lb,int rb,func &EA,func &EB,vector<int> &F){
    if(la>ra)return inf;
    Reduce(EA,v[la],v[ra]);   
    Reduce(EB,v[lb],v[rb]);
    for(cs int &x:F){
        Reduce(Fc[x],v[min(la,lb)],v[max(ra,rb)]);
    }
    if(la==ra) return calc_row_min(la,lb,rb,EA,EB,F).first;
    if(F.size()==0){
        double mn1=EA.querymn(v[la],v[ra]),mn2=EB.querymn(v[lb],v[rb]);
        return mn1+mn2;
    }
    int ma=(la+ra)/2;
    pair<double,ll> res=calc_row_min(ma,lb,rb,EA,EB,F);
    double val=res.fi;ll mb=res.se;
    //Update NewEA and NewEB
    vector<int> SA,SB;
    func NA,NB;
    for(cs int &x:F){
        if(Fc[x].calc(v[ma])<=Fc[x].calc(v[mb]))SA.pb(x);
        else SB.pb(x);
    }
    merge(NA,EA,SA,ma+1,ra);
    merge(NB,EB,SB,lb,mb);
    Anstype ans=Anstype(val,v[ma],mb);
    Anstype ans1=divide(la,ma-1,lb,mb,EA,NB,SA);
    Anstype ans2=divide(ma+1,ra,mb,rb,NA,EB,SB);
    return min(ans,min(ans1,ans2));
}

void solve(vector<vector<double>> s){
    for(cs vector<double> &x:s)
    for(cs double &val:x)v[++n]=val;
    //random_shuffle(v+1,v+n+1);
    cout<<n<<endl;
    auto start = std::chrono::high_resolution_clock::now();
    sort(v+1,v+n+1);
    n=unique(v+1,v+n+1)-v-1;
    for(int i=1;i<=n;i++)id[v[i]]=i;
    vector<int> F;
    for(cs vector<double> &x:s){
        F.pb(Fc.size());
        Fc.pb(build_func(x));
    }
    func x,y;
    Anstype Ans=divide(1,n,1,n,x,y,F);
    auto end = std::chrono::high_resolution_clock::now();
    FILE* ofile = fopen("cpp_result.txt", "w");
    fprintf(ofile, "%.10lf %.10lf\n",Ans.ans,((double)(end-start).count()/1000000000));
    fclose(ofile);
}

/*
in this version 
value range can be large

*/

signed main(){

    //input format:
    //n:number of pointset
    //n lines:mi:number of point and mi number of coordinate
    //freopen("tmp_data.txt","r",stdin);
    int n,k;
    scanf("%d%d",&n,&k);
    vector<vector<double>>s;

    srand(time(NULL));
    for(int i=1;i<=k;i++){
        vector<double>now;
        int ct=0;scanf("%d",&ct);
        for(int j=1;j<=ct;j++){
            double x;
            scanf("%lf",&x);
            now.pb(x);
        }
        sort(now.begin(),now.end());
        s.pb(now);
    }
    solve(s);
}