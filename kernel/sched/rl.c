#include "rl.h"

static const s8 nn_nice_values[OUTPUT_SIZE] = { -5,-4,-3,-2,-1,0,1,2,3,4,5 };

static inline q16_16 q_mul(q16_16 a, q16_16 b) { return (q16_16)(((s64)a * (s64)b) >> Q); }
static inline q16_16 q_add_sat(q16_16 a, q16_16 b)
{
    s64 s = (s64)a + (s64)b;
    if (s > 0x7fffffffLL) s = 0x7fffffffLL;
    if (s < (s64)0x80000000LL) s = 0x80000000LL;
    return (q16_16)s;
}

static const q16_16 W_WAIT_MS_Q = (q16_16)(3 << Q);
static const q16_16 W_RUN_MS_Q  = (q16_16)(-1 << Q);
static const q16_16 W_TA_MS_Q   = (q16_16)(2 << Q);

static inline q16_16 ns_to_ms_q(s64 ns)
{
    s64 ms = ns / 1000000LL;
    s64 q  = ms << Q;
    if (q > 0x7fffffffLL) q = 0x7fffffffLL;
    if (q < (s64)0x80000000LL) q = 0x80000000LL;
    return (q16_16)q;
}

int rl_policy_decide(q16_16 s0, q16_16 s1)
{
    q16_16 state[INPUT_SIZE] = { s0, s1 };

    forward_prop(state);  

    int best = 0;
    q16_16 best_val = nn_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (nn_output[i] > best_val) { best_val = nn_output[i]; best = i; }
    }

    return nn_nice_values[best];  
}

void rl_policy_reward(q16_16 s0, q16_16 s1,
                             int action_idx,
                             s64 wait_ns, s64 run_ns, s64 turnaround_ns)
{
    q16_16 state[INPUT_SIZE] = { s0, s1 };

    q16_16 wait_ms_q = ns_to_ms_q(wait_ns);
    q16_16 run_ms_q  = ns_to_ms_q(run_ns);
    q16_16 ta_ms_q   = ns_to_ms_q(turnaround_ns);

    q16_16 cost_q = 0;
    cost_q = q_add_sat(cost_q, q_mul(W_WAIT_MS_Q, wait_ms_q));
    cost_q = q_add_sat(cost_q, q_mul(W_RUN_MS_Q,  run_ms_q));
    cost_q = q_add_sat(cost_q, q_mul(W_TA_MS_Q,   ta_ms_q));

    q16_16 reward_q = -cost_q;

    nn_back_prop(state, action_idx, reward_q);
}

