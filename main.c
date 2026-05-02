#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "dataset.h"
#include "matrix.h"
#include "network.h"


int main(void) {
    // Set random seed.
    srand(time(NULL));

    // Load data.
    // Dataset train = load_data("train.csv", TRAIN_SIZE);
    // printf("Train samples: %d\n", train.n);
    // assert(train.n == TRAIN_SIZE);
    Dataset test = load_data("test.csv", TEST_SIZE);
    printf("Test samples: %d\n", test.n);
    assert(test.n == TEST_SIZE);


    // === Forward Prop Test w/ batching ===
    Matrix test_batch = wrap_matrix(test.x, 10, IMAGE_SIZE);
    Network network = create_network(10, IMAGE_SIZE);
    init_network(&network);
    forward_propagation(&test_batch, &network);

    printf(" IMAGE |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    | Sum %% |\n");
    for (int batch = 0; batch < 10; batch++) {
        printf(" %02d    |", batch + 1);
        
        float sum = 0.0f;
        for (int neuron = 0; neuron < 10; neuron++) {
            float pred = network.A3.data[batch * 10 + neuron];
            printf(" %5.5f |", pred);
            sum += pred;
        }
        printf(" %5.3f |\n", sum);
    }
    
    free_network(&network);
    // free_data(&train);
    free_data(&test);
}