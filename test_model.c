#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "include/MNIST_data_loader.h"
#include "include/cnn_components.h"
#include "include/model_saver.h"

int main(){

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

    printf("saved model loaded\n");
    load_model(linput);
    printf("saved model loaded\n");
    char test_images_path[] = "data/t10k-images-idx3-ubyte";
    char test_labels_path[] = "data/t10k-labels-idx1-ubyte";

    IdxFile* test_images = get_data(test_images_path);
    printf("test images loaded\n");
    IdxFile* test_labels = get_data(test_labels_path);
    printf("test labels loaded\n");

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

    remove_layer(linput);
    remove_layer(lconv1);
    remove_layer(lconv2);
    remove_layer(lfull1);
    remove_layer(lfull2);
    remove_layer(loutput);

    return 0;
}