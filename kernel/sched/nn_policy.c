#include "rl.h"            /* q16_16, INPUT_SIZE, OUTPUT_SIZE, nn_output */
#include <linux/types.h>

/* Q16.16 fixed-point */
#define Q      16
#define ONE_Q  ((q16_16)1 << Q)

/* Keep hidden layer size & all internal details private to this file */
#define HIDDEN_LAYER_1_SIZE  15

/* 32-bit clamp limits */
#define Q16_16_MAX  ((s64)0x7fffffff)
#define Q16_16_MIN  ((s64)0x80000000)

/* Learning/baseline (private) */
static q16_16 baseline_q = 0;
static const q16_16 BASELINE_DECAY_Q = 58982; /* ~0.9  * 2^16 */
static const q16_16 BASELINE_GAIN_Q  =  6554; /* ~0.1  * 2^16 */
static const q16_16 LR_Q             =    66; /* ~0.001 * 2^16 */

/* Parameters (private) */
static q16_16 W1[HIDDEN_LAYER_1_SIZE * INPUT_SIZE]  = {0};
static q16_16 B1[HIDDEN_LAYER_1_SIZE]               = {0};
static q16_16 W2[OUTPUT_SIZE * HIDDEN_LAYER_1_SIZE] = {0};
static q16_16 B2[OUTPUT_SIZE]                       = {0};

/* Activations (private) */
static q16_16 Z1[HIDDEN_LAYER_1_SIZE] = {0};
static q16_16 Z2[OUTPUT_SIZE]         = {0};

/* Public probability buffer: must have external linkage (matches rl.h extern) */
q16_16 nn_output[OUTPUT_SIZE] = {0};

/* ---- Fixed-point helpers (private) ---- */
static inline q16_16 q_mul(q16_16 a, q16_16 b) { return (q16_16)(((s64)a * (s64)b) >> Q); }
static inline q16_16 q_shl(q16_16 x, int n)    { return (n >= 0) ? (x << n) : (x >> -n); }
static inline q16_16 q_add_sat(q16_16 a, q16_16 b)
{
    s64 s = (s64)a + (s64)b;
    if (s > Q16_16_MAX) s = Q16_16_MAX;
    if (s < Q16_16_MIN) s = Q16_16_MIN;
    return (q16_16)s;
}
static inline q16_16 q_mul_sat(q16_16 a, q16_16 b)
{
    s64 p = ((s64)a * (s64)b) >> Q;
    if (p > Q16_16_MAX) p = Q16_16_MAX;
    if (p < Q16_16_MIN) p = Q16_16_MIN;
    return (q16_16)p;
}

/* ---- exp/softmax (private) ---- */
static const q16_16 INV_LN2_Q = 94603; // 1/ln(2)
static const q16_16 C1 = 45426, C2 = 15739, C3 = 3640, C4 = 630;

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

/* ---- core ops (private) ---- */
static inline void dot_prod(const q16_16 *x, const q16_16 *W, const q16_16 *b,
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

static inline void ReLU(q16_16 *y, unsigned int len)
{
    for (unsigned int i = 0; i < len; i++)
        y[i] = (y[i] > 0) ? y[i] : 0;
}

/* ---- public forward (declared in rl.h) ---- */
void forward_prop(const q16_16 *x)
{
    dot_prod(x,  W1, B1, Z1, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
    ReLU(Z1, HIDDEN_LAYER_1_SIZE);
    dot_prod(Z1, W2, B2, Z2, OUTPUT_SIZE, HIDDEN_LAYER_1_SIZE);
    softmax_q16(Z2, nn_output, OUTPUT_SIZE);
}

/* ---- (optional) learning code stays private; not in rl.h yet ---- */
static inline void update_bias(q16_16 *b, const q16_16 *grad_b, unsigned int len)
{
    for (unsigned int i = 0; i < len; i++) {
        q16_16 step = q_mul_sat(LR_Q, grad_b[i]);
        b[i] = q_add_sat(b[i], step);
    }
}
static inline void update_weights(q16_16 *W, const q16_16 *grad, const q16_16 *input,
                                  unsigned int rows, unsigned int cols)
{
    for (unsigned int r = 0; r < rows; r++) {
        for (unsigned int c = 0; c < cols; c++) {
            q16_16 g = q_mul_sat(grad[r], input[c]);
            q16_16 step = q_mul_sat(LR_Q, g);
            W[r * cols + c] = q_add_sat(W[r * cols + c], step);
        }
    }
}
static inline void back_prop(const q16_16 *state, int action_idx, q16_16 reward_q)
{
    baseline_q = q_add_sat(q_mul(baseline_q, BASELINE_DECAY_Q),
                           q_mul(reward_q, BASELINE_GAIN_Q));
    q16_16 advantage_q = reward_q - baseline_q;

    q16_16 grad_z2[OUTPUT_SIZE];
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++) {
        grad_z2[i] = nn_output[i];
        if ((int)i == action_idx) grad_z2[i] -= ONE_Q;
        grad_z2[i] = q_mul_sat(grad_z2[i], advantage_q);
    }

    q16_16 grad_a1[HIDDEN_LAYER_1_SIZE];
    for (unsigned int h = 0; h < HIDDEN_LAYER_1_SIZE; h++) {
        s64 acc = 0;
        for (unsigned int o = 0; o < OUTPUT_SIZE; o++)
            acc += ((s64)W2[o * HIDDEN_LAYER_1_SIZE + h] * grad_z2[o]) >> Q;
        grad_a1[h] = (q16_16)((acc > Q16_16_MAX) ? Q16_16_MAX :
                              (acc < Q16_16_MIN) ? Q16_16_MIN : acc);
    }

    q16_16 grad_z1[HIDDEN_LAYER_1_SIZE];
    for (unsigned int h = 0; h < HIDDEN_LAYER_1_SIZE; h++)
        grad_z1[h] = (Z1[h] > 0) ? grad_a1[h] : 0;

    update_bias(B2, grad_z2, OUTPUT_SIZE);
    update_weights(W2, grad_z2, Z1, OUTPUT_SIZE, HIDDEN_LAYER_1_SIZE);

    update_bias(B1, grad_z1, HIDDEN_LAYER_1_SIZE);
    update_weights(W1, grad_z1, state, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
}

void nn_back_prop(const q16_16 *state, int action_idx, q16_16 reward_q)
{
    /* If rewards can be delayed (and activations stale), you MAY re-run:
       forward_prop(state);
       before backprop. For now we keep it minimal. */
    back_prop(state, action_idx, reward_q);
}

