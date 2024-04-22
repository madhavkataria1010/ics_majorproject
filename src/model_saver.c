#include <stdio.h>
#include <stdlib.h>
#include "../include/model_saver.h"
#include "../include/cnn_components.h"

// Function to save the model in txt file
void save_model(layer_component *linput)
{
    FILE *file = fopen("results/model.txt", "a");

    if (file == NULL)
    {
        printf("Error: Unable to open file \"model.txt\"\n");
        exit(1);
    }

    // Save the weights and biases of each layer
    layer_component *cur_layer = linput->next_layer;
    while (cur_layer != NULL)
    {
        // saving the weights of current layer
        for (int i = 0; i < cur_layer->num_weights; i++)
        {
            fprintf(file, "%lf\n", cur_layer->weights[i]);
        }
        // saving the biases of current layer
        for (int i = 0; i < cur_layer->num_biases; i++)
        {
            fprintf(file, "%lf\n", cur_layer->biases[i]);
        }
        cur_layer = cur_layer->next_layer;
    }
    fclose(file);
}

// Function to load the model from txt file
void load_model(layer_component *linput)
{

    FILE *file = fopen("results/model.txt", "r");

    if (file == NULL)
    {
        printf("Error: Unable to open file \"model.txt\"\n");
        exit(1);
    }
    // Load the weights and biases of each layer
    layer_component *cur_layer = linput->next_layer;

    // Read the weights and biases from the file
    while (cur_layer != NULL)
    {
        for (int i = 0; i < cur_layer->num_weights; i++)
        {
            fscanf(file, "%lf", &cur_layer->weights[i]);
        }
        for (int i = 0; i < cur_layer->num_biases; i++)
        {
            fscanf(file, "%lf", &cur_layer->biases[i]);
        }
        cur_layer = cur_layer->next_layer;
    }
    fclose(file);
}

// Function to initialize the model architecture
void init_model_architecture(layer_component **linput, layer_component **lconv1, layer_component **lconv2,
                             layer_component **lfull1, layer_component **lfull2, layer_component **loutput)
{
    // Input layer - 1x28x28.
    *linput = create_input_layer(1, 28);
    // Conv1 layer - 16x14x14, 3x3 conv, padding=3, stride=1.
    *lconv1 = create_conv_layer(*linput, 16, 14, 3, 1, 2, 0.1);
    // Conv2 layer - 32x7x7, 7x7 conv, padding=3, stride=1.
    *lconv2 = create_conv_layer(*lconv1, 32, 7, 3, 1, 2, 0.1);
    // FC1 layer - 200 nodes.
    *lfull1 = create_full_layer(*lconv2, 200, 0.1);
    // FC2 layer - 200 nodes.
    *lfull2 = create_full_layer(*lfull1, 200, 0.1); // Fully connected layer - 200 nodes.
    *loutput = create_full_layer(*lfull2, 10, 0.1); // Output layer - 10 nodes.
}
