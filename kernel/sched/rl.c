#include "rl.h"

static const s8 nn_nice_values[OUTPUT_SIZE] = { -5,-4,-3,-2,-1,0,1,2,3,4,5 };

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

