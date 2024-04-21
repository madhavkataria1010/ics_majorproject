#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "include/MNIST_data_loader.h"
#include "include/cnn_components.h"
#include "include/model_saver.h"

int compute_acc(IdxFile* images, IdxFile* labels, layer_component* linput, layer_component* loutput){
    int total = images->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < total; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        get_images(images, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(labels, i);
        /* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
    }
    return ncorrect;
}



double compute_mse_loss(IdxFile* images, IdxFile* labels, layer_component* linput, layer_component* loutput){
    int total = images->dims[0];
    double mse_loss = 0.0;
    for (int i = 0; i < total; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        get_images(images, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(labels, i);
        
        // Convert label to one-hot encoding
        double expected_output[10] = {0};
        expected_output[label] = 1.0;
        
        // Calculate MSE loss
        for (int j = 0; j < 10; j++) {
            double diff = expected_output[j] - y[j];
            mse_loss += diff * diff;
        }
    }
    // Divide by the number of samples to get mean loss
    mse_loss /= total;
    
    return mse_loss;
}

void compute_acc_err(IdxFile* images, IdxFile* labels, layer_component* linput, layer_component* loutput, float* arracc, double* arrerr){
    int total = images->dims[0];
    int ncorrect = 0;
    double MSE = 0;
    for (int i = 0; i < total; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        get_images(images, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(labels, i);
        /* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
        // Convert label to one-hot encoding
        double expected_output[10] = {0};
        expected_output[label] = 1.0;
        
        // Calculate MSE loss
        for (int j = 0; j < 10; j++) {
            double diff = expected_output[j] - y[j];
            MSE += diff * diff;
        }
    }
    MSE/=total;
    float finacc = (float) ncorrect/total*100;

    *arracc = finacc;
    *arrerr = MSE;
}


int main(){
    char train_images_path[] = "data/train-images-idx3-ubyte";
    char train_labels_path[] = "data/train-labels-idx1-ubyte";

    char test_images_path[] = "data/t10k-images-idx3-ubyte";
    char test_labels_path[] = "data/t10k-labels-idx1-ubyte";

    
    /* Use a fixed random seed for debugging. */
    srand(0);

    FILE* trainacc = fopen("trainacc.txt", "a");
    FILE* testacc = fopen("testacc.txt", "a");
    FILE* trainerror = fopen("trainerror.txt", "a");
    FILE* testerror = fopen("testerror.txt", "a");

    IdxFile* train_images = get_data(train_images_path);
    IdxFile* train_labels = get_data(train_labels_path);

    IdxFile* test_images = get_data(test_images_path);
    IdxFile* test_labels = get_data(test_labels_path);

    layer_component *linput, *lconv1, *lconv2, *lfull1, *lfull2, *loutput;
    
    // Initialize model architecture
    init_model_architecture(&linput, &lconv1, &lconv2, &lfull1, &lfull2, &loutput);

    printf("training started\n");
    double rate = 0.1;
    double etotal = 0;
    int nepoch = 10;
    int batch_size = 32;
    int train_size = train_images->dims[0];

    // printf("%d\n", train_size);

    int limit = nepoch * train_size;         
    for (int i = 0; i < limit; i++) {
        /* Pick a random sample from the training data */
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        get_images(train_images, index, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        
        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(train_labels, index);
        
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        
        learn_output(loutput, y);

        etotal += get_total_error(loutput);
        float cur_error = get_total_error(loutput);

        if ((i % batch_size) == 0) {
            /* Minibatch: update the network for every n samples. */
            update_parameters(loutput, rate/batch_size);
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
            etotal = 0;
        }
        if ((i%60000) == 0){
            float accu[2];
            double mse[2];
            compute_acc_err(train_images, train_labels, linput, loutput, &accu[0], &mse[0]);
            compute_acc_err(test_images, test_labels, linput, loutput, &accu[1], &mse[1]);
            fprintf(trainacc, "%f\n", accu[0]);
            fprintf(testacc, "%f\n", accu[1]);
            fprintf(trainerror, "%lf\n", mse[0]);
            fprintf(testerror, "%lf\n", mse[1]);
        }
    }

    delete_data(test_images);
    delete_data(test_labels);
    delete_data(train_images);
    delete_data(train_labels);

    save_model(linput);

    remove_layer(linput);
    remove_layer(lconv1);
    remove_layer(lconv2);
    remove_layer(lfull1);
    remove_layer(lfull2);
    remove_layer(loutput);
    
    fclose(trainacc);
    fclose(testacc);
    fclose(trainerror);
    fclose(testerror);

    return 0;
}
