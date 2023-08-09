# Snippets

- **CP Main**

```cpp
#include <bits/stdc++.h>

int main() {
    std::ios::sync_with_stdio(0); std::cin.tie(0);
    // freopen("", "r", stdin);
    // freopen("", "w", stdout);
}
```
- **Segment Tree**

```cpp
template<typename T> struct segment_tree {
    std::vector<T> SEG; int N; T DEF; T (*OP)(T, T); // Variables
    segment_tree(int n, T d = NULL, T (*c)(T, T) = [](T a, T b) { return a+b; })
        : N(n), DEF(d), OP(c) { SEG.assign(2*N, DEF); } // Constructors

    void pull(int p) { SEG[p] = OP(SEG[2*p], SEG[2*p+1]); } // Mutators
    void update(int p, T v) { if (p < 0 || p >= N) return;
        SEG[p += N] = v; for (p /= 2; p; p /= 2) pull(p); }

    T get(int p) { return (p < 0 || p >= N ? DEF : SEG[p+N]); } // Accessors
    void print() { for (T elem : SEG) std::cout << elem << " "; std::cout << "\n"; }
    T query(int l, int r) { T ra = DEF, rb = DEF; // [l, r]
        if (l < 0 || r < 0 || l >= N || r >= N || l > r) return DEF;
        for (l += N, r += N+1; l < r; l /= 2, r /= 2) { if (l & 1) ra = OP(ra, SEG[l++]);
        if (r & 1) rb = OP(SEG[--r], rb); } return OP(ra, rb); }
};
```

- **Disjoint Set Union**

```cpp
template<typename T> struct disjoint_set {
    std::unordered_map<T, T> TREE; std::unordered_map<T, int> SIZE; // Variables
    disjoint_set(std::vector<T> V = {}) { for (T x : V) add(x); } // Constructors

    bool add(T x) { return TREE.count(x) ? 0 : (TREE[x] = x, SIZE[x] = 1); } // Mutators
    bool unite(T x, T y) { x = get(x), y = get(y); if (x == y) return 0; // Union by size
        if (SIZE[x] < SIZE[y]) std::swap(x, y); TREE[y] = x; SIZE[x] += SIZE[y]; return 1; }

    int size(T x) { return SIZE[get(x)]; } bool same(T x, T y) { return get(x) == get(y); } // Accessors
    int count() { return CNT; } T get(T x) { return TREE[x] == x ? x : TREE[x] = get(TREE[x]); }
    void print() { for (auto [k, v] : TREE) std::printf("%d -> %d (%d)\n", k, get(k), size(k)); }
};
```

- **Combinatoric Utilities**

```cpp
template<typename T> struct combo_cache {
    std::vector<T> INV, IFT, FCT, DRG; T MXN, MOD; // Variables
    combo_cache(T n = 1e6, T m = 1e9+7) : MXN(n), MOD(m) { init(); } // Constructors

    void init() { INV.assign(MXN + 1, 0); INV[0] = INV[1] = 1; // Mutators
        for (T i = 2; i <= MXN; i++) INV[i] = (MOD - MOD/i) * INV[MOD%i] % MOD;
        IFT.assign(MXN + 1, 0); IFT[0] = IFT[1] = 1;
        for (T i = 1; i <= MXN; i++) IFT[i] = IFT[i-1] * INV[i] % MOD;
        FCT.assign(MXN + 1, 0); FCT[0] = FCT[1] = 1;
        for (T i = 1; i <= MXN; i++) FCT[i] = FCT[i-1] * i % MOD;
        DRG.assign(MXN + 1, 0); DRG[0] = 1; DRG[1] = 0;
        for (T i = 2; i <= MXN; i++) DRG[i] = (i-1) * (DRG[i-1] + DRG[i-2]) % MOD; }

    T inv(T n) { return (n >= 0 && n <= MXN ? INV[n] : -1); } // Accessors
    T ift(T n) { return (n >= 0 && n <= MXN ? IFT[n] : -1); }
    T fct(T n) { return (n >= 0 && n <= MXN ? FCT[n] : -1); }
    T drg(T n) { return (n >= 0 && n <= MXN ? DRG[n] : -1); }
    T cat(T n) { return (n >= 0 && n <= MXN/2 ? bin(2*n, n) * inv(n+1) % MOD : -1); }
    T bin(T n, T k) { return (k < n ? fct(n) * ift(k) % MOD * ift(n-k) % MOD : -1); }
    T str(T n, T k) { return bin(n + k - 1, n); }
};
```

- **Ordered Set**

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

template <typename T> using ordered_set = __gnu_pbds::tree<
    T, __gnu_pbds::null_type, std::less<T>, __gnu_pbds::rb_tree_tag,
    __gnu_pbds::tree_order_statistics_node_update // find_by_order, order_of_key
>;
```

- **Generic Trie**

```cpp
template<typename T, typename TL> struct generic_trie {
    struct node { int CNT, END; std::unordered_map<T, node*> NXT; node() : CNT(0), END(0) {} } *HEAD;
    generic_trie() : HEAD(new node()) { } // Variables & Constructors

    void insert(TL val) { node *curr = HEAD; curr->CNT++; for (T v : val) { // Mutators
        if (!curr->NXT.count(v)) curr->NXT[v] = new node();
        curr = curr->NXT[v]; curr->CNT++; } curr->END = 1; }

    bool search(TL val) { node *curr = HEAD; for (T v : val) { // Accessors
        if (!curr->NXT.count(v)) return 0; curr = curr->NXT[v]; } return curr->END; }
    int count(TL val) { node *curr = HEAD; for (T v : val) {
        if (!curr->NXT.count(v)) return 0; curr = curr->NXT[v]; } return curr->CNT; }
};
```

- **Topological Sort**
```cpp
template<typename T> struct topo_sort {
    std::vector<std::vector<T>> ADJ; std::vector<T> ANS, VIS; T N; // Variables
    topo_sort(T n, std::vector<std::vector<T>> &adj) : ADJ(adj), N(n) { } // Constructors

    void dfs(T v) { VIS[v] = 1; for (T u : ADJ[v]) if (!VIS[u]) { dfs(u); } ANS.push_back(v); } // Accessors
    std::vector<T> sort() { ANS.clear(); VIS.assign(N, 0); for (T i = 0; i < N; i++) {
        if (!VIS[i]) { dfs(i); } } std::reverse(ANS.begin(), ANS.end()); return ANS; }
    bool check() { std::map<T, T> MP; for (T i = 0; i < N; i++) MP[ANS[i]] = i; for (T i = 0; i < N; i++) {
        for (T q = 0; q < ADJ[i].size(); q++) if (MP[i] > MP[ADJ[i][q]]) return false; } return true; }
};
```

- **Least Common Ancestor**
```cpp
template<typename T> struct segment_lca { // Variables & Constructors
    std::vector<std::vector<T>> ADJ; std::vector<T> HGT, FST, VIS, EUL, SEG; T N, P, R;
    segment_lca(T n, std::vector<std::vector<T>> &adj, T r = 0) : ADJ(adj), N(n), P(0) {
        HGT.resize(N); FST.resize(N); VIS.resize(N); EUL.resize(2*N); dfs(r, 0);
        SEG.resize(4*N); for (T i = 0; i < 2*N; i++) update(i, EUL[i]); }

    void pull(T p) { SEG[p] = comb(SEG[2*p], SEG[2*p+1]); } // Mutators
    void update(T p, T v) { SEG[p += 2*N] = v; for (p /= 2; p; p /= 2) pull(p); }
    void dfs(T v, T h) { VIS[v] = 1; HGT[v] = h; FST[v] = P; EUL[P++] = v;
        for (T u : ADJ[v]) if (!VIS[u]) { dfs(u, h+1); EUL[P++] = v; } }

    T comb(T a, T b) { return HGT[a] < HGT[b] ? a : b; } // Helpers & Accessors
    T dist(T a, T b) { return HGT[a] + HGT[b] - 2*HGT[find(a, b)]; }
    T find(T a, T b) { if (FST[a] > FST[b]) std::swap(a, b); return query(FST[a], FST[b]); }
    T query(T l, T r) { T ra = SEG[l += 2*N], rb = SEG[r += 2*N]; for (r++; l < r; l /= 2, r /= 2)
        { if (l & 1) ra = comb(ra, SEG[l++]); if (r & 1) rb = comb(SEG[--r], rb); } return comb(ra, rb); }
};
```

- **Sparse Table**

```cpp
template<typename T> struct sparse_table {
    std::vector<std::vector<T>> ST; int N, K; T (*OP)(T, T); // Variables & Constructors
    sparse_table(int n, std::vector<T> &V, T (*c)(T, T) = [](T a, T b) { return std::min(a, b); }) : N(n), OP(c) {
        K = log2(N); ST.assign(K+1, std::vector<T>(N)); std::copy(V.begin(), V.end(), ST[0].begin());
        for (int k = 1; k <= K; k++) for (int i = 0; i + (1 << k) <= N; i++)
        ST[k][i] = OP(ST[k-1][i], ST[k-1][i+(1<<(k-1))]); }

    int log2(int n) { return 31 - __builtin_clz(n); } // Helpers & Accessors
    void print() { for (auto V : ST) { for (T v : V) std::cout << v << " "; std::cout << "\n"; } }
    T idem(int l, int r) { int k = log2(r-l+1); return OP(ST[k][l], ST[k][r-(1<<k)+1]); } // [l, r]
    T query(int l, int r, int d = 0) { T val = d; for (int i = K; i >= 0; i--) // [l, r]
        if (1<<i <= r - l + 1) { val = OP(val, ST[i][l]); l += (1<<i); } return val; }
};
```

- **Point Geometry**

```cpp
template <typename T> struct vec { // Variables & Constructors
    T X, Y, Z; vec() : X(0), Y(0), Z(0) { } vec(T x) : X(x), Y(0), Z(0) { }
    vec(T x, T y) : X(x), Y(y), Z(0) { } vec(T x, T y, T z) : X(x), Y(y), Z(z) { }
    void read2() { std::cin >> X >> Y; } void read3() { std::cin >> X >> Y >> Z; }

    vec operator *(const T &c) { return vec(X * c, Y * c, Z * c); } // Mutators
    vec operator /(const T &c) { return vec(X / c, Y / c, Z / c); }
    vec operator +(const vec &p) { return vec(X + p.X, Y + p.Y, Z + p.Z); }
    vec operator -(const vec &p) { return vec(X - p.X, Y - p.Y, Z - p.Z); }
    bool operator <(const vec &p) const { return std::tie(X, Y) < std::tie(p.X, p.Y); }

    T norm() { return X*X + Y*Y + Z*Z; } // ||X + Y|| < ||X|| + ||Y||, |X * Y| < ||X|| * ||Y||
    T dot(vec &p) { return X*p.X + Y*p.Y + Z*p.Z; } // cos(theta), X x Y = 0 if X || Y
    T cross2(const vec &p) { return X*p.Y - Y*p.X; } // RHR, > 0 if p CCW of this, < 0 if p CW of this
    T tri2(vec &p, vec &q) { return (p - *this).cross2(q - *this); } // 2 * area of triangle, < 0 if p CCW q
    vec cross3(vec &p) { return vec(Y*p.Z - Z*p.Y, Z*p.X - X*p.Z, X*p.Y - Y*p.X); } // sin(theta), X x Y = -Y x X
};
```

- **Binary Lifting**

```cpp
template<typename T> struct binary_lift { // Variables & Constructors
    std::vector<std::vector<T>> ADJ, UP; std::vector<T> IN, OUT; T N, L, C;
    binary_lift(T n, std::vector<std::vector<T>> &adj, T r = 0) : ADJ(adj), N(n), L(log2(N)), C(0) {
        IN.resize(N); OUT.resize(N); UP.resize(N, std::vector<T>(L+1, -1)); dfs(r, -1); }
    void dfs(T v, T p) { IN[v] = C++; UP[v][0] = p; for (T i = 1; i <= L; i++) { if (UP[v][i-1] != -1)
        UP[v][i] = UP[UP[v][i-1]][i-1]; } for (T u : ADJ[v]) { if (u != p) dfs(u, v); } OUT[v] = C++; }

    T log2(T n) { return 31 - __builtin_clz(n); } // Helpers & Accessors
    bool anc(T v, T u) { return IN[v] <= IN[u] && OUT[v] >= OUT[u]; }
    T jump(T v, T k) { for (T i = 0; i <= L; i++) { if (k & (1 << i)) { if ((v = UP[v][i]) == -1) break; } } return v; }
    T lca(T v, T u) { if (anc(v, u)) return v; if (anc(u, v)) return u; for (T i = L; i >= 0; i--)
        { if (!anc(UP[v][i], u)) v = UP[v][i]; } return UP[v][0]; }
};
```

- **Heavy Light Decomposition**

```cpp
template<typename T> struct heavy_decomp { // Variables & Constructors
    std::vector<std::vector<T>> ADJ; std::vector<T> PAR, DEP, HVY, POS, VAL, SEG, HD; T N, P; T (*OP)(T, T);
    heavy_decomp(T n, std::vector<std::vector<T>> &adj, std::vector<T> &val, T (*op)(T, T), T r = 0) {
        N = n; P = 0; ADJ = adj; VAL = val; OP = op; PAR.resize(N); DEP.resize(N); HVY.assign(N, -1); POS.resize(N);
        HD.resize(N); dfs(r); decomp(r, r); SEG.resize(2*N); for (T i = 0; i < N; i++) update(POS[i], VAL[i]); }

    void pull(T p) { SEG[p] = OP(SEG[2*p], SEG[2*p+1]); } // Mutators
    void update(T p, T v) { SEG[p += N] = v; for (p /= 2; p; p /= 2) pull(p); }
    int dfs(T v) { T sz = 1, mx = 0; for (T u : ADJ[v]) if (u != PAR[v]) { PAR[u] = v; DEP[u] = DEP[v] + 1;
        T usz = dfs(u); sz += usz; if (usz > mx) { mx = usz; HVY[v] = u; } } return sz; }
    void decomp(T v, T h) { POS[v] = P++; HD[v] = h; if (HVY[v] != -1) decomp(HVY[v], h);
        for (T u : ADJ[v]) if (u != PAR[v] && u != HVY[v]) decomp(u, u); }

    T query(T l, T r, T d = 0) { T ra = d, rb = d; for (l += N, r += N+1; l < r; l /= 2, r /= 2) { // Accessors
        if (l & 1) ra = OP(ra, SEG[l++]); if (r & 1) rb = OP(SEG[--r], rb); } return OP(ra, rb); }
    T calc(T a, T b, T d = 0) { T res = d; for (; HD[a] != HD[b]; b = PAR[HD[b]]) {
        if (DEP[HD[a]] > DEP[HD[b]]) std::swap(a, b); res = OP(res, query(POS[HD[b]], POS[b], d)); }
        if (DEP[a] > DEP[b]) std::swap(a, b); return OP(res, query(POS[a], POS[b], d)); }
};
```
