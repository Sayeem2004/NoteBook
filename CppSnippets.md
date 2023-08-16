# Snippets

- **CP Main** Generic competitive programming main function. Contains some utilities for faster I/O and file redirection.

```cpp
#include <bits/stdc++.h>

int main(int ARGC, char* ARGV[]) {
    std::ios::sync_with_stdio(0); std::cin.tie(0);
    // freopen("", "r", stdin);
    // freopen("", "w", stdout);
}
```

- **Program Timer** Used to time the execution of a program. No parameters needed.

```cpp
struct timer { clock_t S, E, CPS = CLOCKS_PER_SEC; timer() { }; void start() { S = clock(); }
    void end() { E = clock(); std::cout << "Time Elapsed: " << (E-S)/double(CPS) << "s\n"; } };
```

- **Segment Tree** Used for range queries and updates in O(log N) time. Parameters needed are a node type in the tree, the number of nodes, a default value for nodes, and a function to combine nodes.

```cpp
template<typename T> struct segment_tree {
    std::vector<T> SEG; int N; T DEF; T (*OP)(T, T); // Variables
    segment_tree(int N, T D = NULL, T (*OP)(T, T) = [](T A, T B) { return A + B; })
        : N(N), DEF(D), OP(OP) { SEG.assign(2 * N, DEF); } // Constructors

    void pull(int P) { SEG[P] = OP(SEG[2 * P], SEG[2 * P + 1]); } // Mutators
    void update(int P, T V) { if (P < 0 || P >= N) return; SEG[P += N] = V; for (P /= 2; P; P /= 2) pull(P); }

    T get(int P) { return (P < 0 || P >= N ? DEF : SEG[P + N]); } // Accessors
    void print() { for (T E : SEG) std::cout << E << " "; std::cout << "\n"; }
    T query(int L, int R) { T RA = DEF, RB = DEF; if (L < 0 || R >= N || L > R) return DEF; // [L, R]
        for (L += N, R += N+1; L < R; L /= 2, R /= 2) { if (L & 1) RA = OP(RA, SEG[L++]);
        if (R & 1) RB = OP(SEG[--R], RB); } return OP(RA, RB); }
};
```

- **Disjoint Set Union** Used for maintaining sets of elements and merging them in O(1) time. Able to retrieve size of a set, number of sets, and whether two elements are in the same set in O(1) time. Parameters needed are a node type in the disjoint sets, and a vector of nodes to initialize the disjoint set with.

```cpp
template<typename T> struct disjoint_set {
    std::unordered_map<T, T> TREE; std::unordered_map<T, int> SIZE; int CNT; // Variables
    disjoint_set() { } disjoint_set(std::vector<T> &V) { for (T X : V) add(X); } // Constructors

    bool add(T X) { return TREE.count(X) ? 0 : (CNT++, TREE[X] = X, SIZE[X] = 1); } // Mutators
    bool unite(T X, T Y) { X = get(X), Y = get(Y); if (X == Y) return 0; CNT--; // Union by size
        if (SIZE[X] < SIZE[Y]) std::swap(X, Y); TREE[Y] = X; SIZE[X] += SIZE[Y]; return 1; }

    int size(T X) { return SIZE[get(X)]; } bool same(T X, T Y) { return get(X) == get(Y); } // Accessors
    int count() { return CNT; } T get(T X) { return TREE[X] == X ? X : TREE[X] = get(TREE[X]); }
    void print() { for (auto [K, V] : TREE) std::printf("%d -> %d (%d)\n", K, get(K), size(K)); }
};
```

- **Combinatoric Utilities** Used for O(N) precalculation and O(1) query of factorial, inverses, derangements, catalan numbers, and related concepts. Parameters needed are the maximum number to precalculate, and the modulo to use.

```cpp
template<typename T> struct combo_cache {
    std::vector<T> INV, IFT, FCT, DRG; T MXN, MOD; // Variables
    combo_cache(T MXN = 1e6, T MOD = 1e9 + 7) : MXN(MXN), MOD(MOD) { // Constructors
        INV.assign(MXN + 1, 0); INV[0] = 1; INV[1] = 1; IFT.assign(MXN + 1, 0); IFT[0] = 1; IFT[1] = 1;
        FCT.assign(MXN + 1, 0); FCT[0] = 1; FCT[1] = 1; DRG.assign(MXN + 1, 0); DRG[0] = 1; DRG[1] = 0;
        for (T i = 2; i <= MXN; i++) { FCT[i] = FCT[i - 1] * i % MOD; INV[i] = (MOD - MOD / i) * INV[MOD % i] % MOD;
        IFT[i] = IFT[i - 1] * INV[i] % MOD; DRG[i] = (i - 1) * (DRG[i - 1] + DRG[i - 2]) % MOD; } }

    T str(T N, T K) { return bin(N + K - 1, N); } // Accessors
    T bin(T N, T K) { return (K < N ? fct(N) * ift(K) % MOD * ift(N - K) % MOD : -1); }
    T cat(T N) { return (N >= 0 && N <= MXN / 2 ? bin(2 * N, N) * inv(N + 1) % MOD : -1); }
    T pow(T N, T K) { T R = 1; while (K > 0) { if (K & 1) R = R * N % MOD; N = N * N % MOD; K >>= 1; } return R; }
    T inv(T N) { return (N >= 0 && N <= MXN ? INV[N] : -1); } T ift(T N) { return (N >= 0 && N <= MXN ? IFT[N] : -1); }
    T fct(T N) { return (N >= 0 && N <= MXN ? FCT[N] : -1); } T drg(T N) { return (N >= 0 && N <= MXN ? DRG[N] : -1); }
};
```

- **Ordered Set** Used for O(log N) insertion and deletion, along with O(log N) querying of position. Parameters needed are the data type inserted into the set.

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

template <typename T> using ordered_set = __gnu_pbds::tree< T, __gnu_pbds::null_type, std::less<T>,
    __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update >; // find_by_order, order_of_key
```

- **Topological Sort** Used to sort a DAG in topological order, which means that for every edge (u, v), u comes before v in the ordering. Parameters needed are the number of nodes and an adjacency list.

```cpp
template<typename T> struct topo_sort {
    std::vector<std::vector<T>> ADJ; std::vector<T> ANS, VIS; T N; // Variables
    topo_sort(T N, std::vector<std::vector<T>> &ADJ) : N(N), ADJ(ADJ) { } // Constructors

    void dfs(T V) { VIS[V] = 1; for (T U : ADJ[V]) if (!VIS[U]) { dfs(U); } ANS.push_back(V); } // Accessors
    std::vector<T> sort() { ANS.clear(); VIS.assign(N, 0); for (T i = 0; i < N; i++) {
        if (!VIS[i]) { dfs(i); } } std::reverse(ANS.begin(), ANS.end()); return ANS; }
    bool check() { std::map<T, T> MP; for (T i = 0; i < N; i++) MP[ANS[i]] = i; for (T i = 0; i < N; i++) {
        for (T q = 0; q < ADJ[i].size(); q++) if (MP[i] > MP[ADJ[i][q]]) return false; } return true; }
};
```

- **Least Common Ancestor** Used to find a node that is an ancestor of two nodes in a tree, and is as low as possible. This allows easy distance operations as well. Parameters needed are the number of nodes, an adjacency list, and a root node.

```cpp
template<typename T> struct segment_lca { // Variables & Constructors
    std::vector<std::vector<T>> ADJ; std::vector<T> HGT, FST, VIS, EUL, SEG; T N, P, R;
    segment_lca(T N, std::vector<std::vector<T>> &ADJ, T R = 0) : N(N), ADJ(ADJ), P(0) {
        HGT.resize(N); FST.resize(N); VIS.resize(N); EUL.resize(2 * N); dfs(R, 0);
        SEG.resize(4 * N); for (T i = 0; i < 2 * N; i++) update(i, EUL[i]); }

    void pull(T P) { SEG[P] = comb(SEG[2 * P], SEG[2 * P + 1]); } // Mutators
    void update(T P, T V) { SEG[P += 2 * N] = V; for (P /= 2; P; P /= 2) pull(P); }
    void dfs(T V, T H) { VIS[V] = 1; HGT[V] = H; FST[V] = P; EUL[P++] = V;
        for (T U : ADJ[V]) if (!VIS[U]) { dfs(U, H + 1); EUL[P++] = V; } }

    T comb(T A, T B) { return HGT[A] < HGT[B] ? A : B; } // Accessors
    T dist(T A, T B) { return HGT[A] + HGT[B] - 2 * HGT[find(A, B)]; }
    T find(T A, T B) { if (FST[A] > FST[B]) std::swap(A, B); return query(FST[A], FST[B]); }
    T query(T L, T R) { T RA = SEG[L += 2 * N], RB = SEG[R += 2 * N]; for (R++; L < R; L /= 2, R /= 2)
        { if (L & 1) RA = comb(RA, SEG[L++]); if (R & 1) RB = comb(SEG[--R], RB); } return comb(RA, RB); }
};
```

- **Sparse Table** Used to answer static range queries in O(log N) time, and O(1) time is possible if the combination operation is idempotent, which means that OP(OP(X, X), OP(X, X)) = OP(X, X), basically overlap of intervals doesn't affect answer. Parameters needed are the number of nodes, a vector of nodes, and a combination function.

```cpp
template<typename T> struct sparse_table {
    std::vector<std::vector<T>> ST; int N, K; T (*OP)(T, T); // Variables & Constructors
    sparse_table(int N, std::vector<T> &V, T (*OP)(T, T) = [](T A, T B) { return std::min(A, B); }) : N(N), OP(OP) {
        K = log2(N); ST.assign(K + 1, std::vector<T>(N)); std::copy(V.begin(), V.end(), ST[0].begin());
        for (int k = 1; k <= K; k++) for (int i = 0; i + (1 << k) <= N; i++)
        ST[k][i] = OP(ST[k - 1][i], ST[k - 1][i + (1 << (k - 1))]); }

    int log2(int N) { return 31 - __builtin_clz(N); } // Helpers & Accessors
    void print() { for (auto VEC : ST) { for (T V : VEC) std::cout << V << " "; std::cout << "\n"; } }
    T idem(int L, int R) { int K = log2(R - L + 1); return OP(ST[K][L], ST[K][R - (1 << K) + 1]); } // [l, r]
    T query(int L, int R, int D = 0) { T V = D; for (int i = K; i >= 0; i--) // [l, r]
        if (1 << i <= R - L + 1) { V = OP(V, ST[i][L]); L += (1 << i); } return V; }
};
```

- **Point Geometry** Used for basic point geometry operations, also stores some useful facts about these operations in the comments. Parameters needed are the data type of the coordinates.

```cpp
template <typename T> struct vec { // Variables & Constructors
    T X, Y, Z; vec() : X(0), Y(0), Z(0) { } vec(T X) : X(X), Y(0), Z(0) { }
    vec(T X, T Y) : X(X), Y(Y), Z(0) { } vec(T X, T Y, T Z) : X(X), Y(Y), Z(Z) { }
    void read2() { std::cin >> X >> Y; } void read3() { std::cin >> X >> Y >> Z; }

    vec operator *(const T &C) { return vec(X * C, Y * C, Z * C); } // Mutators
    vec operator /(const T &C) { return vec(X / C, Y / C, Z / C); }
    vec operator +(const vec &P) { return vec(X + P.X, Y + P.Y, Z + P.Z); }
    vec operator -(const vec &P) { return vec(X - P.X, Y - P.Y, Z - P.Z); }
    bool operator <(const vec &P) const { return std::tie(X, Y) < std::tie(P.X, P.Y); }

    T norm() { return X * X + Y * Y + Z * Z; } // ||X + Y|| < ||X|| + ||Y||, |X * Y| < ||X|| * ||Y||
    T dot(vec &P) { return X * P.X + Y * P.Y + Z * P.Z; } // cos(theta), X x Y = 0 if X || Y
    T cross2(const vec &P) { return X * P.Y - Y * P.X; } // RHR, > 0 if p CCW of this, < 0 if p CW of this
    T tri2(vec &P, vec &Q) { return (P - *this).cross2(Q - *this); } // 2 * area of triangle, < 0 if p CCW q
    vec cross3(vec &P) { return vec(Y * P.Z - Z * P.Y, Z * P.X - X * P.Z, X * P.Y - Y * P.X); } // sin, X x Y = -Y x X
};
```

- **Binary Lifting** Used to find the Kth ancestor of a node in a tree in O(log N), which then allows for O(log N) LCA. Parameters needed are the number of nodes, an adjacency list, and a root node.

```cpp
template<typename T> struct binary_lift { // Variables & Constructors
    std::vector<std::vector<T>> ADJ, UP; std::vector<T> IN, OUT; T N, L, C;
    binary_lift(T N, std::vector<std::vector<T>> &ADJ, T R = 0) : N(N), ADJ(ADJ), L(log2(N)), C(0) {
        IN.resize(N); OUT.resize(N); UP.resize(N, std::vector<T>(L + 1, -1)); dfs(R, -1); }
    void dfs(T V, T P) { IN[V] = C++; UP[V][0] = P; for (T i = 1; i <= L; i++) { if (UP[V][i - 1] != -1)
        UP[V][i] = UP[UP[V][i - 1]][i - 1]; } for (T U : ADJ[V]) { if (U != P) dfs(U, V); } OUT[V] = C++; }

    T log2(T N) { return 31 - __builtin_clz(N); } // Helpers & Accessors
    bool anc(T V, T U) { return IN[V] <= IN[U] && OUT[V] >= OUT[U]; }
    T jump(T V, T K) { for (T i = 0; i <= L; i++) { if (K & (1 << i)) { if ((V = UP[V][i]) == -1) break; } } return V; }
    T lca(T V, T U) { if (anc(V, U)) return V; if (anc(U, V)) return U; for (T i = L; i >= 0; i--)
        { if (!anc(UP[V][i], U)) V = UP[V][i]; } return UP[V][0]; }
};
```

- **Heavy Light Decomposition** Used to split a tree into O(log N) paths such that the path from any node to the root is the concatenation of some of these paths. This allows for O(log N) path queries using segment tree. Parameters needed are the number of nodes, an adjacency list, a value for each node, and a combination function.

```cpp
template<typename T> struct heavy_decomp { // Variables & Constructors
    std::vector<std::vector<T>> ADJ; std::vector<T> PAR, DEP, HVY, POS, VAL, SEG, HD; T N, P; T (*OP)(T, T);
    heavy_decomp(T N, std::vector<std::vector<T>> &ADJ, std::vector<T> &VAL, T (*OP)(T, T), T R = 0) {
        N = N; P = 0; ADJ = ADJ; VAL = VAL; OP = OP; PAR.resize(N); DEP.resize(N); HVY.assign(N, -1); POS.resize(N);
        HD.resize(N); dfs(R); decomp(R, R); SEG.resize(2 * N); for (T i = 0; i < N; i++) update(POS[i], VAL[i]); }

    void pull(T P) { SEG[P] = OP(SEG[2 * P], SEG[2 * P + 1]); } // Mutators
    void update(T P, T V) { SEG[P += N] = V; for (P /= 2; P; P /= 2) pull(P); }
    int dfs(T V) { T SZ = 1, MX = 0; for (T U : ADJ[V]) if (U != PAR[V]) { PAR[U] = V; DEP[U] = DEP[V] + 1;
        T USZ = dfs(U); SZ += USZ; if (USZ > MX) { MX = USZ; HVY[V] = U; } } return SZ; }
    void decomp(T V, T H) { POS[V] = P++; HD[V] = H; if (HVY[V] != -1) decomp(HVY[V], H);
        for (T U : ADJ[V]) if (U != PAR[V] && U != HVY[V]) decomp(U, U); }

    T query(T L, T R, T D = 0) { T RA = D, RB = D; for (L += N, R += N + 1; L < R; L /= 2, R /= 2) { // Accessors
        if (L & 1) RA = OP(RA, SEG[L++]); if (R & 1) RB = OP(SEG[--R], RB); } return OP(RA, RB); }
    T calc(T A, T B, T D = 0) { T RES = D; for (; HD[A] != HD[B]; B = PAR[HD[B]]) {
        if (DEP[HD[A]] > DEP[HD[B]]) std::swap(A, B); RES = OP(RES, query(POS[HD[B]], POS[B], D)); }
        if (DEP[A] > DEP[B]) std::swap(A, B); return OP(RES, query(POS[A], POS[B], D)); }
};
```

- **Matrix Struct** Used for matrix operations in O(N^3) time. Parameters needed are the data type of the matrix, row size, column size, default value, and modulo. Uses binary exponentiation to speed up matrix exponentiation.

```cpp
template<typename T> struct matrix { // Variables & Constructors
    T R, C, D, MOD; std::vector<std::vector<T>> M; matrix(T N) : matrix(N, N) { }
    matrix(T R, T C, T D = 0, T MOD = 1e9+7) : R(R), C(C), D(D), MOD(MOD), M(R, std::vector<T>(C, D)) { }
    matrix(std::vector<std::vector<T>> M, T MOD = 1e9+7) : R(M.size()), C(M[0].size()), MOD(MOD), M(M) { }

    matrix operator +(const matrix &O) const { matrix A(R, C); for (T i = 0; i < R; i++) // Operators
        for (T q = 0; q < C; q++) A.M[i][q] = (M[i][q] + O.M[i][q]) % MOD; return A; }
    matrix operator -(const matrix &O) const { matrix A(R, C); for (T i = 0; i < R; i++)
        for (T q = 0; q < C; q++) A.M[i][q] = (M[i][q] - O.M[i][q]) % MOD; return A; }
    matrix operator *(const matrix &O) const { matrix A(R, O.C); for (T i = 0; i < R; i++) for (T q = 0; q < O.C; q++)
        for (T r = 0; r < C; r++) A.M[i][r] = (A.M[i][r] + M[i][q] * O.M[q][r]) % MOD; return A; }
    matrix operator ^(const T &P) const { if (!P) { matrix A(R); for (T i = 0; i < R; i++) A.M[i][i] = 1; return A; }
        matrix A = *this ^ (P >> 1); A *= A; if (P & 1) A *= *this; return A; }
    std::vector<T> operator &(const std::vector<T> &V) const { std::vector<T> A(R); for (T i = 0; i < R; i++)
        for (T q = 0; q < C; q++) A[i] = (A[i] + M[i][q] * V[q]) % MOD; return A; }

    matrix operator +=(const matrix &O) { return *this = *this + O; } // Assignments
    matrix operator -=(const matrix &O) { return *this = *this - O; }
    matrix operator *=(const matrix &O) { return *this = *this * O; }
    matrix operator ^=(const T &P) { return *this = *this ^ P; }
    void print() { for (T i = 0; i < R; i++) for (T q = 0; q < C; q++) std::cout << M[i][q] << " \n"[q == C - 1]; }
};
```

- **Number Cache** Used to precalculate prime factors and the mobius function in O(N log N) time, and then allows for O(1) access to these values. Also contains some useful inverse, power, and overflow functions. Parameters needed are the maximum number to precalculate, and the modulo to use.

```cpp
template<typename T> struct num_cache {
    std::vector<T> LPF, MOB; T MXN, MOD; // Variables
    num_cache(T MXN = 1e6, T MOD = 1e9+7) : MXN(MXN), MOD(MOD) { // Constructors
        LPF.assign(MXN + 1, 0); MOB.assign(MXN + 1, 1); for (int i = 2; i <= MXN; i++) {
        if (!LPF[i]) { for (int q = i; q <= MXN; q += i) if (!LPF[q]) LPF[q] = i; }
        if (LPF[i/LPF[i]] == LPF[i]) MOB[i] = 0; else MOB[i] = -MOB[i/LPF[i]]; } }

    T lpf(T N) { return (N >= 0 && N <= MXN ? LPF[N] : -1); } // Accessors
    T mob(T N) { return (N >= 0 && N <= MXN ? MOB[N] : -2); }
    T inv(T X) { return (X == 1 ? 1 : (MOD - MOD / X) * inv(MOD % X) % MOD); }
    T pow(T N, T K) { T R = 1; while (K > 0) { if (K & 1) R = R * N % MOD; N = N * N % MOD; K >>= 1; } return R; }

    bool aover(T A, T B) { return __builtin_add_overflow_p(A, B, (T) 0); } // Helpers
    bool sover(T A, T B) { return __builtin_sub_overflow_p(A, B, (T) 0); }
    bool mover(T A, T B) { return __builtin_mul_overflow_p(A, B, (T) 0); }
    bool dover(T A, T B) { return __builtin_div_overflow_p(A, B, (T) 0); }
};
```
