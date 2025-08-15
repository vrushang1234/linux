#include <linux/types.h>

// Q16.16 fixed-point
typedef s32 q16_16;
#define Q      16
#define ONE_Q  ((q16_16)1 << Q)

#define INPUT_SIZE           2
#define HIDDEN_LAYER_1_SIZE  15
#define OUTPUT_SIZE          11

// 32-bit clamp limits
#define Q16_16_MAX  ((s64)0x7fffffff)
#define Q16_16_MIN  ((s64)0x80000000)

// Flattened row-major weights: W[row * cols + col]
static q16_16 W1[HIDDEN_LAYER_1_SIZE * INPUT_SIZE]  = {0};
static q16_16 B1[HIDDEN_LAYER_1_SIZE]               = {0};
static q16_16 W2[OUTPUT_SIZE * HIDDEN_LAYER_1_SIZE] = {0};
static q16_16 B2[OUTPUT_SIZE]                       = {0};

static q16_16 Z1[HIDDEN_LAYER_1_SIZE] = {0};
static q16_16 Z2[OUTPUT_SIZE]         = {0};
static q16_16 output[OUTPUT_SIZE]     = {0};

static inline q16_16 q_mul(q16_16 a, q16_16 b) { return (q16_16)(((s64)a * (s64)b) >> Q); }
static inline q16_16 q_shl(q16_16 x, int n)    { return (n >= 0) ? (x << n) : (x >> -n); }

// constants in Q16.16
static const q16_16 INV_LN2_Q = 94603; // 1/ln(2)

// Poly for 2^f on f∈[0,1)
static const q16_16 C1 = 45426;  // 0.69314718 * 2^16
static const q16_16 C2 = 15739;  // 0.24022651 * 2^16
static const q16_16 C3 =  3640;  // 0.05550411 * 2^16
static const q16_16 C4 =   630;  // 0.00961813 * 2^16

static inline q16_16 exp2_frac_q(q16_16 f)
{
    q16_16 t = q_mul(C4, f);
    t = q_mul(t + C3, f);
    t = q_mul(t + C2, f);
    t = q_mul(t + C1, f);
    return t + ONE_Q;
}

static inline q16_16 fast_exp2_q(q16_16 y)
{
    s32 k = y >> Q;
    q16_16 f = y - (k << Q);
    if (f < 0) { f += ONE_Q; k -= 1; }

    q16_16 frac = exp2_frac_q(f);

    if (k <= -31) return 0;
    if (k >=  31) return 0x7fffffff;
    return q_shl(frac, k);
}

static inline q16_16 fast_exp_q(q16_16 x)
{
    q16_16 y = q_mul(x, INV_LN2_Q);
    if (y < (q16_16)(-30 << Q)) y = (q16_16)(-30 << Q);
    if (y > (q16_16)( 30 << Q)) y = (q16_16)( 30 << Q);
    return fast_exp2_q(y);
}

static inline void ReLU(q16_16 *y, unsigned int len)
{
    for (unsigned int i = 0; i < len; i++)
        y[i] = (y[i] > 0) ? y[i] : 0;
}

static inline void softmax_q16(const q16_16 *x, q16_16 *p, unsigned int len)
{
    if (!len) return;

    q16_16 xmax = x[0];
    for (unsigned int i = 1; i < len; i++)
        if (x[i] > xmax) xmax = x[i];

    u64 sum = 0;
    for (unsigned int i = 0; i < len; i++) {
        q16_16 xi = x[i] - xmax;
        if (xi < (q16_16)(-16 << Q)) xi = (q16_16)(-16 << Q);
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
        u64 num = ((u64)(u32)fast_exp_q(xi)) << Q;
        p[i] = (q16_16)(num / sum);
    }
}

// Generic matrix-vector multiply for flat row-major W
static inline void dot_prod(const q16_16 *x,
                              const q16_16 *W,
                              const q16_16 *b,
                              q16_16 *z,
                              unsigned int rows,
                              unsigned int cols)
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

static inline void forward_prop(const q16_16 *x)
{
    dot_prod(x,  W1, B1, Z1, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
    ReLU(Z1, HIDDEN_LAYER_1_SIZE);
    dot_prod(Z1, W2, B2, Z2, OUTPUT_SIZE, HIDDEN_LAYER_1_SIZE);
    softmax_q16(Z2, output, OUTPUT_SIZE);
}

