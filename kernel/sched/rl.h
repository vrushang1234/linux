#ifndef RL_H
#define RL_H

#include <linux/types.h>   /* s32, etc. */

/* Fixed-point type used by the policy */
typedef s32 q16_16;

/* Keep these consistent across files */
#define INPUT_SIZE   2
#define OUTPUT_SIZE  11

/* Exposed by nn_policy.c */
extern void forward_prop(const q16_16 *x);
/* nn_policy.c must define this with external linkage (no 'static') */
extern q16_16 nn_output[OUTPUT_SIZE];

/* RL API: decide-only */
int rl_policy_decide(q16_16 s0, q16_16 s1);

#endif /* RL_H */

