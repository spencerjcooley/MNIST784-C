#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"


int main(void) {
    Dataset train = load_data("train.csv", TRAIN_SIZE);
    Dataset test = load_data("test.csv", TEST_SIZE);

    printf("Train samples: %d\n", train.n);
    printf("Test samples: %d\n", test.n);
    
    free_data(&train);
    free_data(&test);
}