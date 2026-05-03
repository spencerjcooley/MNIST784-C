#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "dataset.h"
#include "matrix.h"
#include "network.h"
#include "train.h"



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
    size_t offset = 0; // Batch index.
    Matrix test_batch_x = wrap_matrix(&test.x[offset], 10, IMAGE_SIZE);
    int *test_batch_y = &test.y[offset];
    
    Network network = create_network(10, IMAGE_SIZE);
    init_network(&network);
    forward_propagation(&test_batch_x, &network);

    printf(" IMAGE |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    | Sum %% |\n");
    for (int batch = 0; batch < 10; batch++) {
        printf(" %02d-%d  |", batch + 1, test_batch_y[batch]);
        
        float sum = 0.0f;
        for (int neuron = 0; neuron < 10; neuron++) {
            float pred = network.A3.data[batch * 10 + neuron];
            printf(" %5.5f |", pred);
            sum += pred;
        }
        printf(" %5.3f |\n", sum);
    }

    // === Loss Test ===
    float loss = cce_loss(&network.Z3, test_batch_y);
    printf("\nCCE Loss: %f\n", loss);

    // === Back Prop Test ===
    backward_propagation(&network, &test_batch_x, test_batch_y, 0.01);
    forward_propagation(&test_batch_x, &network);

    printf(" IMAGE |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    | Sum %% |\n");
    for (int batch = 0; batch < 10; batch++) {
        printf(" %02d-%d  |", batch + 1, test_batch_y[batch]);
        
        float sum = 0.0f;
        for (int neuron = 0; neuron < 10; neuron++) {
            float pred = network.A3.data[batch * 10 + neuron];
            printf(" %5.5f |", pred);
            sum += pred;
        }
        printf(" %5.3f |\n", sum);
    }
    loss = cce_loss(&network.Z3, test_batch_y);
    printf("\nCCE Loss: %f\n", loss);
    

    // === Memory ===
    free_network(&network);
    // free_data(&train);
    free_data(&test);
}