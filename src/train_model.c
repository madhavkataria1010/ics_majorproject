#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/MNIST_data_loader.h"
#include "../include/cnn_components.h"
#include "../include/model_saver.h"

// Accuary function
int compute_acc(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput)
{
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
    }
    return ncorrect;
}

// Mean Squared Error los function
double compute_mse_loss(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput)
{
    int total = images->dims[0];
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
    }
    // Divide by the number of samples to get mean loss
    mse_loss /= total; // averaging the loss

    return mse_loss;
}

void compute_acc_err(IdxFile *images, IdxFile *labels, layer_component *linput, layer_component *loutput, float *arracc, double *arrerr)
{
    int total = images->dims[0];    // total number of images
    int ncorrect = 0;               // number of correct predictions
    double MSE = 0;                 // mean squared error loss
    for (int i = 0; i < total; i++) // for each image
    {
        uint8_t img[28 * 28];
        double x[28 * 28];
        double y[10];
        get_images(images, i, img); // storing the image in img
        for (int j = 0; j < 28 * 28; j++)
        {
            x[j] = img[j] / 255.0; // storing the image in x
        }
        set_input_layer(linput, x); // setting the input layer
        get_output(loutput, y);     // getting the output by model in y
        int label = get_labels(labels, i);
        /* Pick the most probable label. */
        int mj = -1; // mj is the index of the maximum value of y
        for (int j = 0; j < 10; j++)
        {
            if (mj < 0 || y[mj] < y[j])
            {
                mj = j;
            } // finding the index of the maximum value of y
        }
        if (mj == label)
        {
            ncorrect++; // incrementing the correct count
        }
        // Convert label to one-hot encoding
        double expected_output[10] = {0};
        expected_output[label] = 1.0;

        // Calculate MSE loss
        for (int j = 0; j < 10; j++)
        {
            double diff = expected_output[j] - y[j]; // calculating the difference between the expected output and the output by model
            MSE += diff * diff;
        }
    }
    MSE /= total;
    float finacc = (float)ncorrect / total * 100;

    *arracc = finacc;
    *arrerr = MSE; // storing the accuracy and the mean squared error loss in array acc and err
}

int main()
{
    char train_images_path[] = "data/train-images-idx3-ubyte";
    char train_labels_path[] = "data/train-labels-idx1-ubyte";

    char test_images_path[] = "data/t10k-images-idx3-ubyte";
    char test_labels_path[] = "data/t10k-labels-idx1-ubyte";

    // Seed random number generator of the model
    srand(0);

    FILE *trainacc = fopen("results/trainacc.txt", "a");
    FILE *testacc = fopen("results/testacc.txt", "a");
    FILE *trainerror = fopen("results/trainerror.txt", "a");
    FILE *testerror = fopen("results/testerror.txt", "a"); // opening the files to store the accuracy and the mean squared error loss

    IdxFile *train_images = get_data(train_images_path);
    IdxFile *train_labels = get_data(train_labels_path);

    IdxFile *test_images = get_data(test_images_path);
    IdxFile *test_labels = get_data(test_labels_path); // getting the data from the files

    layer_component *linput, *lconv1, *lconv2, *lfull1, *lfull2, *loutput; // defining the layers

    // Initialize model architecture
    init_model_architecture(&linput, &lconv1, &lconv2, &lfull1, &lfull2, &loutput); // initializing the model architecture

    printf("training started\n");
    double learning_rate = 0.01; // learning rate
    double total_error = 0;
    int num_epoch = 10;                     // number of epochs
    int batch_size = 32;                    // batch size
    int train_size = train_images->dims[0]; // size of the training data

    // printf("%d\n", train_size);

    int limit = num_epoch * train_size;
    for (int i = 0; i < limit; i++)
    {
        uint8_t img[28 * 28];
        double x[28 * 28];
        double y[10];
        int index = rand() % train_size;      // generating a random index
        get_images(train_images, index, img); // storing the image in img
        for (int j = 0; j < 28 * 28; j++)
        {
            x[j] = img[j] / 255.0; // storing the image in x
        }

        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(train_labels, index); // getting the output by model in y

        for (int j = 0; j < 10; j++)
        {
            y[j] = (j == label) ? 1 : 0; // setting the value of y to 1 if the label is equal to j
        }

        learn_output(loutput, y);

        total_error += get_total_error(loutput);
        float cur_error = get_total_error(loutput); // getting the total error

        if ((i % batch_size) == 0)
        {
            update_parameters(loutput, learning_rate / batch_size); // updating the parameters
        }
        if ((i % 1000) == 0)
        {
            printf("epoch=%d, i=%d, error=%.4f\n", (i/60000) + 1, i%60000, total_error / 1000);
            total_error = 0;
        }
        if ((i % 60000) == 0)
        {
            printf("--------Epoch -> %i--------\n", (i / 60000)+1);
            float accu[2];
            double mse[2];
            compute_acc_err(train_images, train_labels, linput, loutput, &accu[0], &mse[0]);
            compute_acc_err(test_images, test_labels, linput, loutput, &accu[1], &mse[1]);
            fprintf(trainacc, "%f\n", accu[0]);
            fprintf(testacc,"%f\n", accu[1]); // storing the accuracy and the mean squared error loss in the files
            fprintf(trainerror, "%lf\n", mse[0]);
            fprintf(testerror, "%lf\n", mse[1]); // storing the accuracy and the mean squared error loss in the files
        }
        if (i % 120000 == 0)
            learning_rate /= 10;
    }

    delete_data(test_images);
    delete_data(test_labels);
    delete_data(train_images);
    delete_data(train_labels); // deleting the train/test data

    save_model(linput);

    remove_layer(linput);
    remove_layer(lconv1);
    remove_layer(lconv2);
    remove_layer(lfull1); // removing the layers
    remove_layer(lfull2);
    remove_layer(loutput);

    fclose(trainacc);
    fclose(testacc);
    fclose(trainerror); // closing the files
    fclose(testerror);

    return 0;
}
