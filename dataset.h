#ifndef DATASET_H
#define DATASET_H

#define IMAGE_SIZE 784
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define MAX_LINE_LENGTH 4096 // 4kb, well over max line length.

typedef struct {
    float *x;
    int *y;
    int n;
} Dataset;

Dataset load_data(const char *filename, int n_samples);
void free_data(Dataset *dataset);

#endif