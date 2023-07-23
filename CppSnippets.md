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
template<class T> struct segment_tree {
    std::vector<T> SEG; int N; T DEF; T (*OP)(T, T); // Variables
    segment_tree(int n, T d = (T) NULL, T (*c)(T, T) = [](T a, T b) { return a + b; })
        { N = n; DEF = d; OP = c; SEG.assign(2 * N, DEF); } // Constructor

    void pull(int p) { SEG[p] = OP(SEG[2 * p], SEG[2 * p + 1]); } // Mutators
    void update(int p, T v) { if (p < 0 || p >= N) return;
        SEG[p += N] = v; for (p /= 2; p; p /= 2) pull(p); }

    T get(int p) { return (p < 0 || p >= N ? DEF : SEG[p + N]); } // Accessors
    void print() { for (T elem : SEG) std::cout << elem << " "; std::cout << "\n"; }
    T query(int l, int r) { T ra = DEF, rb = DEF; // [l, r]
        if (l < 0 || r < 0 || l >= N || r >= N || l > r) return DEF;
        for (l += N, r += N + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) ra = OP(ra, SEG[l++]);
            if (r & 1) rb = OP(SEG[--r], rb);
        } return OP(ra, rb); }
};
```

- **Disjoint Set Union**

```cpp
template<class T> struct disjoint_set {
    std::unordered_map<T, T> TREE; std::unordered_map<T, int> SIZE; // Variables
    disjoint_set(std::vector<T> V = {}) { for (T x : V) add(x); } // Constructor

    bool add(T x) { return TREE.count(x) ? 0 : (TREE[x] = x, SIZE[x] = 1); } // Mutators
    bool unite(T x, T y) { x = get(x), y = get(y); // Union by size
        if (x == y) return 0; if (SIZE[x] < SIZE[y]) swap(x, y);
        TREE[y] = x; SIZE[x] += SIZE[y]; return 1; }

    int size(T x) { return SIZE[get(x)]; } // Accessors
    bool same(T x, T y) { return get(x) == get(y); }
    T get(T x) { return TREE[x] == x ? x : TREE[x] = get(TREE[x]); }
    void print() { for (auto i = TREE.begin(); i != TREE.end(); i++)
        std::printf("%d -> {%d} (%d)\n", i->first, get(i->first), size(i->first)); }
};
```

- **Combinatoric Utilities**

```cpp
template<class T> struct combo_cache {
    std::vector<T> INV, IFT, FCT, DRG; T MXN = 1e6, MOD = 1e9+7; // Variables
    combo_cache(T n = 1e6, T m = 1e9+7) { MXN = n; MOD = m; init(); } // Constructor

    void init() { INV.assign(MXN + 1, 0); INV[0] = INV[1] = 1; // Initializer
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
    T bin (T n, T k) { return (k < n ? fct(n) * ift(k) % MOD * ift(n-k) % MOD : -1); }
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

- **Numeric Utilities**

```cpp
template<class T> struct num_util {
    T gcd(T a, T b) { return b ? gcd(b, a % b) : a; }
    T lcm(T a, T b) { return a / gcd(a, b) * b; }
    T pow(T a, T b, T m) { if (!b) return 1; T r = pow(a*a%m, b/2, m); return b&1 ? r*a%m : r; }
};
```
