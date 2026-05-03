#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "dataset.h"
#include "matrix.h"
#include "network.h"
#include "train.h"

#define BATCH_SIZE 25
#define EPOCHS 20
#define LEARNING_RATE 0.01



int main(void) {
    // Set random seed.
    srand(time(NULL));

    // === Load Data ===
    Dataset train_set = load_data("train.csv", TRAIN_SIZE);
    printf("Training samples: %d\n", train_set.n);
    assert(train_set.n == TRAIN_SIZE);
    Dataset test_set = load_data("test.csv", TEST_SIZE);
    printf("Testing samples: %d\n", test_set.n);
    assert(test_set.n == TEST_SIZE);

    // === Full Training ===
    Network network = create_network(BATCH_SIZE, IMAGE_SIZE);
    init_network(&network);
    train(&network, &train_set, BATCH_SIZE, EPOCHS, LEARNING_RATE);

    // === Testing ===
    test(&network, &test_set, BATCH_SIZE);

    // === Memory ===
    free_network(&network);
    free_data(&train_set);
    free_data(&test_set);
}