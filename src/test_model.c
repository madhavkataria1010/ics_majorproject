#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/MNIST_data_loader.h"
#include "../include/cnn_components.h"
#include "../include/model_saver.h"

int compute_acc(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput);
double compute_mse_loss(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput);

int main()
{
    // Load the saved model
    layer_component *linput, *lconv1, *lconv2, *lfull1, *lfull2, *loutput;

    // Initialize model architecture
    init_model_architecture(&linput, &lconv1, &lconv2, &lfull1, &lfull2, &loutput);

    load_model(linput);

    printf("saved model loaded\n");
    // Load test data
    char test_images_path[] = "data/t10k-images-idx3-ubyte";
    char test_labels_path[] = "data/t10k-labels-idx1-ubyte";

    IdxFile *test_images = get_data(test_images_path);
    printf("test images loaded\n");
    IdxFile *test_labels = get_data(test_labels_path);
    printf("test labels loaded\n");

    printf("testing\n");
    // calculate accuracy and loss
    int ncorrect = compute_acc(test_images, test_labels, linput, loutput);
    float accuracy = (float)ncorrect / test_images->dims[0];

    double MSE_loss = compute_mse_loss(test_images, test_labels, linput, loutput);

    printf("Accuracy: %f\n", accuracy);
    printf("Mean Squared Error Loss: %f\n", MSE_loss);

    delete_data(test_images);
    delete_data(test_labels);

    remove_layer(linput);
    remove_layer(lconv1);
    remove_layer(lconv2);
    remove_layer(lfull1);
    remove_layer(lfull2);
    remove_layer(loutput);

    return 0;
}

// Accuary function
int compute_acc(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput)
{   
    printf("computing accuracy\n");
    int total = images->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < total; i++) // for each image
    {
        uint8_t img[28 * 28];
        double x[28 * 28];
        double y[10];
        get_images(images, i, img); // storing the image in img and then in x
        for (int j = 0; j < 28 * 28; j++)
        {
            x[j] = img[j] / 255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y); // getting the output by model in y
        int label = get_labels(labels, i);
        // finding the value of y with maximum value
        int mj = -1; // mj is the index of the maximum value of y
        for (int j = 0; j < 10; j++)
        {
            if (mj < 0 || y[mj] < y[j])
            {
                mj = j;
            }
        }
        if (mj == label)
        {
            ncorrect++; // incrementing the correct count
        }
        if (i%1000 == 0)
        {
            printf("images processed: %d\n", i);
        }
    }
    return ncorrect;
}

// Mean Squared Error los function
double compute_mse_loss(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput)
{
    int total = images->dims[0];
    printf("computing loss\n");
    double mse_loss = 0.0;
    for (int i = 0; i < total; i++)
    {
        uint8_t img[28 * 28]; // storing the image in img
        double x[28 * 28];    // storing the image in x
        double y[10];
        get_images(images, i, img);
        for (int j = 0; j < 28 * 28; j++)
        {
            x[j] = img[j] / 255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y); // getting the output by model in y
        int label = get_labels(labels, i);

        // Convert label to one-hot encoding
        double expected_output[10] = {0}; // expected output is the one hot encoding of the label
        expected_output[label] = 1.0;     // setting the value of the label to 1

        // Calculate MSE loss
        for (int j = 0; j < 10; j++)
        {
            double diff = expected_output[j] - y[j];
            mse_loss += diff * diff; // calculating the mean squared error loss
        }
        if (i%1000 == 0)
        {
            printf("images processed: %d\n", i);
        }
    }
    // Divide by the number of samples to get mean loss
    mse_loss /= total; // averaging the loss

    return mse_loss;
}
