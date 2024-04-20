#include <stdio.h>
#include <stdlib.h>
#include "include/model_saver.h"

void save_model(layer_component* linput) {
    FILE* file = fopen("model.txt", "a");

    if (file == NULL) {
        printf("Error: Unable to open file \"model.txt\"\n");
        exit(1);
    }

    layer_component* cur_layer = linput->next_layer; 
    while (cur_layer != NULL) {
        for (int i=0; i<cur_layer->num_nodes; i++) {
            fprintf(file, "%f\n", cur_layer->weights[i]);
            fprintf(file, "%f\n", cur_layer->biases[i]);
        }
        cur_layer = cur_layer->next_layer;
    }
    fclose(file);
}

void load_model(layer_component* linput) {

    FILE* file = fopen("model.txt", "r"); 

    if (file == NULL) {
        printf("Error: Unable to open file \"model.txt\"\n");
        exit(1);
    }

    layer_component* cur_layer = linput->next_layer;

    while (cur_layer != NULL) {
        for (int i=0; i<cur_layer->num_nodes; i++) {
            fscanf(file, "%f\n", &cur_layer->weights[i]);
            fscanf(file, "%f\n", &cur_layer->biases[i]);
        }
        cur_layer = cur_layer->next_layer;
    }
    printf("t,");
    fclose(file);
}