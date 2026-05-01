#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

Dataset load_data(const char *filename, int n_samples) {
    Dataset data;
    data.n = n_samples;
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening the file.\n");
        data.x = NULL;
        data.y = NULL;
        data.n = 0;
        return data;
    }

    data.x = malloc(n_samples * IMAGE_SIZE * sizeof(float));
    data.y = malloc(n_samples * sizeof(int));

    if (!data.x || !data.y) {
        perror("Error allocating memory.\n");
        fclose(file);
        data.n = 0;
        return data;
    }

    int n = 0;
    char line[MAX_LINE_LENGTH];

    while (n < n_samples && fgets(line, MAX_LINE_LENGTH, file)) {
        char *token = strtok(line, ",");
        for (int i = 0; i < IMAGE_SIZE; i++) {
            if (!token) break;
            // Normalise pixel values.
            data.x[n * IMAGE_SIZE + i] = atof(token) / 255.0f;
            token = strtok(NULL, ",");
        }

        if (token) {
            data.y[n] = atoi(token);
        }

        n++;
    }

    fclose(file);

    data.n = n;
    return data;
}

void free_data(Dataset *dataset) {
    free(dataset->x);
    free(dataset->y);
    dataset->x = NULL;
    dataset->y = NULL;
    dataset->n = 0;
}