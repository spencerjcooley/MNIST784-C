#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

void dense_forward(
    const Matrix *X,
    const Matrix *W,
    const Matrix *B,
    Matrix *Y
);

#endif