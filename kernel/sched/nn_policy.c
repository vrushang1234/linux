// Minimal fixed-point NN forward pass (Q16.16) with 2 hidden layers (tanh) and softmax output.
// Input:  8 features
// Hidden: 50 (tanh) -> 70 (tanh)
// Output: 11 probabilities (softmax)
// Everything is defined in this file; no headers required.

#include <linux/types.h>
#include "nn_policy.h"

/* ---------- Fixed-point type & basics ---------- */
/* Safe ops */
static inline q16_16 q_shl(q16_16 x, int n) { return (n >= 0) ? (x << n) : (x >> -n); }
static inline q16_16 q_add_sat(q16_16 a, q16_16 b) {
    s64 s = (s64)a + (s64)b;
    if (s > Q16_16_MAX) s = Q16_16_MAX;
    if (s < Q16_16_MIN) s = Q16_16_MIN;
    return (q16_16)s;
}
static inline q16_16 q_mul(q16_16 a, q16_16 b) { return (q16_16)(((s64)a * (s64)b) >> Q); }
static inline q16_16 q_mul_sat(q16_16 a, q16_16 b) {
    s64 p = ((s64)a * (s64)b) >> Q;
    if (p > Q16_16_MAX) p = Q16_16_MAX;
    if (p < Q16_16_MIN) p = Q16_16_MIN;
    return (q16_16)p;
}
static inline q16_16 q_div_sat(q16_16 a, q16_16 b) {
    if (b == 0) return (a >= 0) ? ONE_Q : (q16_16)(-ONE_Q);
    s64 n = ((s64)a) << Q;
    s64 q = n / b;
    if (q > Q16_16_MAX) q = Q16_16_MAX;
    if (q < Q16_16_MIN) q = Q16_16_MIN;
    return (q16_16)q;
}

/* ---------- Sizes ---------- */
#define INPUT_SIZE   8
#define H1_SIZE      50
#define H2_SIZE      70

/* ---------- Parameters (initialized to 0; load externally as needed) ---------- */
static q16_16 W1[H1_SIZE * INPUT_SIZE] = {0};
static q16_16 B1[H1_SIZE]              = {0};

static q16_16 W2[H2_SIZE * H1_SIZE]    = {0};
static q16_16 B2[H2_SIZE]              = {0};

static q16_16 W3[OUTPUT_SIZE * H2_SIZE]= {0};
static q16_16 B3[OUTPUT_SIZE]          = {0};

/* ---------- Activations / scratch ---------- */
static q16_16 Z1[H1_SIZE];
static q16_16 Z2[H2_SIZE];
static q16_16 Z3[OUTPUT_SIZE];


/* ---------- fast exp & softmax (Q16.16) ---------- */
/* exp(x) via exp2(y) with y = x / ln(2). Polynomial for exp2 fractional part. */
static const q16_16 INV_LN2_Q = 94603;    /* ~1/ln(2) * 2^16 */
static const q16_16 C1 = 45426, C2 = 15739, C3 = 3640, C4 = 630;


static inline q16_16 exp2_frac_q(q16_16 f) {
    q16_16 t = q_mul(C4, f);
    t = q_mul(q_add_sat(t, C3), f);
    t = q_mul(q_add_sat(t, C2), f);
    t = q_mul(q_add_sat(t, C1), f);
    return q_add_sat(t, ONE_Q);
}
static inline q16_16 fast_exp2_q(q16_16 y) {
    s32 k = y >> Q;
    q16_16 f = y - (k << Q);
    if (f < 0) { f += ONE_Q; k -= 1; }
    q16_16 frac = exp2_frac_q(f);
    if (k <= -31) return 0;
    if (k >=  31) return (q16_16)0x7fffffff;
    return q_shl(frac, k);
}
static inline q16_16 fast_exp_q(q16_16 x) {
    /* clamp x / ln(2) to keep range sane */
    q16_16 y = q_mul(x, INV_LN2_Q);
    if (y < (q16_16)(-30 << Q)) y = (q16_16)(-30 << Q);
    if (y > (q16_16)( 30 << Q)) y = (q16_16)( 30 << Q);
    return fast_exp2_q(y);
}

static inline void softmax_q16(const q16_16 *x, q16_16 *p, unsigned int len) {
    if (!len) return;

    q16_16 xmax = x[0];
    for (unsigned int i = 1; i < len; i++)
        if (x[i] > xmax) xmax = x[i];

    /* sum exp(x - xmax) in Q16.16, but keep sum in u64 to avoid overflow */
    u64 sum = 0;
    for (unsigned int i = 0; i < len; i++) {
        q16_16 xi = x[i] - xmax;
        if (xi < (q16_16)(-16 << Q)) xi = (q16_16)(-16 << Q); /* tail clamp */
        sum += (u64)(u32)fast_exp_q(xi);
    }

    if (!sum) {
        q16_16 u = (q16_16)(ONE_Q / (int)len);
        for (unsigned int i = 0; i < len; i++) p[i] = u;
        return;
    }

    for (unsigned int i = 0; i < len; i++) {
        q16_16 xi = x[i] - xmax;
        if (xi < (q16_16)(-16 << Q)) xi = (q16_16)(-16 << Q);
        u64 num = ((u64)(u32)fast_exp_q(xi)) << Q; /* scale to Q16.16 */
        p[i] = (q16_16)(num / sum);
    }
}

/* ---------- tanh (Q16.16) ---------- */
/* tanh(x) = (e^{2x} - 1) / (e^{2x} + 1) with clamps to [-1, 1] */
static inline q16_16 tanh_q16_scalar(q16_16 x) {
    /* For large |x|, tanh -> +/-1 quickly; clamp input to keep exp stable */
    if (x >  (q16_16)( 8 << Q)) return  (q16_16) ONE_Q;
    if (x <  (q16_16)(-8 << Q)) return  (q16_16)(-ONE_Q);

    q16_16 two_x = q_shl(x, 1);
    q16_16 e2x   = fast_exp_q(two_x);         /* Q16.16 */
    q16_16 num   = q_add_sat(e2x, (q16_16)(-ONE_Q));
    q16_16 den   = q_add_sat(e2x, ONE_Q);
    q16_16 t     = q_div_sat(num, den);
    /* extra hard clamp to [-1,1] in Q16.16 */
    if (t >  ONE_Q)  t = ONE_Q;
    if (t < -ONE_Q)  t = -ONE_Q;
    return t;
}
static inline void tanh_vec_q16(q16_16 *v, unsigned int len) {
    for (unsigned int i = 0; i < len; i++) v[i] = tanh_q16_scalar(v[i]);
}

/* ---------- Dense (matrix-vector) ---------- */
static inline void dense_mv(const q16_16 *x, const q16_16 *W, const q16_16 *b,
                            q16_16 *z, unsigned int rows, unsigned int cols)
{
    for (unsigned int r = 0; r < rows; r++) {
        s64 acc = (s64)b[r];
        const q16_16 *wr = &W[r * cols];
        for (unsigned int c = 0; c < cols; c++)
            acc += ((s64)wr[c] * (s64)x[c]) >> Q;
        if (acc > Q16_16_MAX) acc = Q16_16_MAX;
        if (acc < Q16_16_MIN) acc = Q16_16_MIN;
        z[r] = (q16_16)acc;
    }
}

/* ---------- Forward pass ---------- */
/* Writes probabilities into global nn_output[OUTPUT_SIZE] */
void forward_prop(q16_16 *x)
{
    /* layer 1: 8 -> 50, tanh */
    dense_mv(x,  W1, B1, Z1, H1_SIZE, INPUT_SIZE);
    tanh_vec_q16(Z1, H1_SIZE);

    /* layer 2: 50 -> 70, tanh */
    dense_mv(Z1, W2, B2, Z2, H2_SIZE, H1_SIZE);
    tanh_vec_q16(Z2, H2_SIZE);

    /* output: 70 -> 11, softmax */
    dense_mv(Z2, W3, B3, Z3, OUTPUT_SIZE, H2_SIZE);
    softmax_q16(Z3, nn_output, OUTPUT_SIZE);
}

