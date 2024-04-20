#ifndef CNN_COMPONENTS_H
#define CNN_COMPONENTS_H

typedef struct layer_component{
    int layer_id;
    struct layer_component* prev_layer;
    struct layer_component* next_layer;

    int depth;
    int dimension;

    int num_nodes;
    double* output;
    double* gradient;
    double* errors;

    int num_biases;
    int num_weights;

    double* biases;
    double* weights;

    double* up_biases;
    double* up_weights;

    int kernel_size;
    int padding;
    int stride;

}layer_component;

layer_component* create_input_layer(int depth, int dimension);

layer_component* create_conv_layer(layer_component* prev_layer, int depth, int dimension,
                                    int kernel_size, int padding, int striding, double std);

layer_component* create_full_layer(layer_component* prev_layer, int num_nodes, double std);

void remove_layer(layer_component* self);

void set_input_layer(layer_component* self, double* values);

void feedforwd_conv(layer_component* self);

void feedforwd_full(layer_component* self);

void get_output(layer_component* self, double* outputs);

double get_total_error(layer_component* self);

void feedback_conv(layer_component* self);

void feedback_full(layer_component* self);

void learn_output(layer_component* self, double* values);

void update_parameters(layer_component* self, double lr);

double rnd();

double norm_rnd();

double tanh(double x);

double tanh_grad(double x);

double relu(double x);

double relu_grad(double x);

#endif