#pragma once

typedef s32 q16_16;           /* signed 16.16 fixed point */
#define Q        16
#define ONE_Q    ((q16_16)1 << Q)

/* Saturation bounds for 32-bit Q16.16 */
#define Q16_16_MAX  ((s64)0x7fffffff)
#define Q16_16_MIN  ((s64)0x80000000)


#define OUTPUT_SIZE  11
extern q16_16 nn_output[OUTPUT_SIZE];
void forward_prop(q16_16 *x);

