#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "MNIST_data_loader.h"
#include "cnn_components.h"
#include "model_saver.h"

int main(){
    char train_images_path[] = "data/train-images-idx3-ubyte";
    char train_labels_path[] = "data/train-labels-idx1-ubyte";

    char test_images_path[] = "data/t10k-images-idx3-ubyte";
    char test_labels_path[] = "data/t10k-labels-idx1-ubyte";

    
    /* Use a fixed random seed for debugging. */
    srand(0);

    // printf("st");
    
    IdxFile* train_images = get_data(train_images_path);
    // printf("tri\n");
    IdxFile* train_labels = get_data(train_labels_path);
    // printf("tr lab\n");

    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    layer_component* linput = create_input_layer(1, 28);
    /* Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. */
    /* (14-1)*2+3 < 28+1*2 */
    layer_component* lconv1 = create_conv_layer(linput, 16, 14, 3, 1, 2, 0.1);
    /* Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2. */
    /* (7-1)*2+3 < 14+1*2 */
    layer_component* lconv2 = create_conv_layer(lconv1, 32, 7, 3, 1, 2, 0.1);
    /* FC1 layer - 200 nodes. */
    layer_component* lfull1 = create_full_layer(lconv2, 200, 0.1);
    /* FC2 layer - 200 nodes. */
    layer_component* lfull2 = create_full_layer(lfull1, 200, 0.1);
    /* Output layer - 10 nodes. */
    layer_component* loutput = create_full_layer(lfull2, 10, 0.1);

    // printf("tt");
    printf("training started\n");
    double rate = 0.1;
    double etotal = 0;
    int nepoch = 2;
    int batch_size = 32;
    int train_size = train_images->dims[0];
    printf("%d\n", train_size);

    int limit = nepoch * train_size;
    for (int i = 0; i < 3; i++) {
                // printf("get_45g5br44\n");
        /* Pick a random sample from the training data */
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        get_images(train_images, index, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }

        for (int p=0; p<28; p++){
            for (int q=0; q<28; q++){
                printf("%d ", x[p*28 + q]);
            }
            printf("\n");
        }
        // printf("get_image\n");
        set_input_layer(linput, x);       // printf("set input\n");
        get_output(loutput, y);      //  printf("get_output\n");
        for (int o=0; o<10; o++)
        printf("%f ",y[o]);
        printf("\n");
        int label = get_labels(train_labels, index);
                //printf("get_label\n");
        printf("%d\n", label);

        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        // printf("learn_output\n");
        learn_output(loutput, y);       // printf("learn output\n");
        // printf("learn_output001\n");
        etotal += get_total_error(loutput);
        printf("%f\n", etotal);
        // if ((i % batch_size) == 0) {
        //     /* Minibatch: update the network for every n samples. */
        //     update_parameters(loutput, rate/batch_size);
        // }
        // if ((i % 1000) == 0) {
        //     fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
        //     etotal = 0;
        // }
        layer_component* sel = linput;
        while (sel!=NULL)
        {
            sel = sel->next_layer;
            for (int z=0; z<sel->num_nodes && z<5; z++)
            printf("%f ", sel->errors[i]);

            printf("\n");
        }
        
    }

    delete_data(train_images);
    delete_data(train_labels);

    IdxFile* test_images = get_data(test_images_path);
    IdxFile* test_labels = get_data(test_labels_path);

    printf("testing\n");
    int ntests = test_images->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < ntests; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        get_images(test_images, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        set_input_layer(linput, x);
        get_output(loutput, y);
        int label = get_labels(test_labels, i);
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
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d\n", i);
        }
    }
    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);

    delete_data(test_images);
    delete_data(test_labels);

    // save_model(linput);

    remove_layer(linput);
    remove_layer(lconv1);
    remove_layer(lconv2);
    remove_layer(lfull1);
    remove_layer(lfull2);
    remove_layer(loutput);

    return 0;
}