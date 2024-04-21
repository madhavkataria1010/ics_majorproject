#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/cnn_components.h"

double rnd()
{
    double output = (double)rand() / RAND_MAX; // generate random number between 0 and 1
    return output;
}

double norm_rnd()
{
    double output = (double)(rnd() + rnd() + rnd() + rnd() - 2) * 1.724; // generate random number between -2 and 2
                                                                         // with mean 0 and variance 1
    return output;
}

double relu(double x)
{
    if (x >= 0)
        return x; // if x is greater than or equal to 0, return x
    else
        return 0; // else return 0
}

double relu_grad(double x)
{
    if (x >= 0)
        return 1; // if x is greater than or equal to 0, return 1
    else
        return 0; // else return 0
}

double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2 * x)) - 1.0; // tanh function     // 2/(1+e^(-2x)) - 1
}

double tanh_grad(double y)
{
    return (double)1 - y * y; // tanh gradient      // 1 - y^2
}

layer_component *create_input_layer(int depth, int dimension)
{
    layer_component *self = (layer_component *)calloc(1, sizeof(layer_component)); // allocate memory for the layer
    self->prev_layer = NULL;
    self->next_layer = NULL; // set the previous and next layers to NULL
    self->layer_id = 0;      // set the layer id to 0

    self->depth = depth; // set all the parameters as entered by the user
    self->dimension = dimension;
    self->type = INPUT_LAYER;

    self->num_nodes = depth * dimension * dimension; // calculate the number of nodes in the layer

    // allocate memory for the output, gradient and errors arrays
    self->output = (double *)calloc(self->num_nodes, sizeof(double));
    self->gradient = (double *)calloc(self->num_nodes, sizeof(double));
    self->errors = (double *)calloc(self->num_nodes, sizeof(double));

    self->num_biases = 0; // for input layer, number of biases and weights are 0
    self->num_weights = 0;

    self->biases = (double *)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double *)calloc(self->num_biases, sizeof(double));

    self->weights = (double *)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double *)calloc(self->num_weights, sizeof(double));

    return self;
}

layer_component *create_conv_layer(layer_component *prev_layer, int depth, int dimension,
                                   int kernel_size, int padding, int stride, double std)
{

    layer_component *self = (layer_component *)calloc(1, sizeof(layer_component)); // allocate memory for the layer

    self->kernel_size = kernel_size; // set all the parameters as entered by the user
    self->padding = padding;
    self->stride = stride;
    self->type = CONV_LAYER;

    self->prev_layer = prev_layer; // set the previous and next layers of prev_layer and self
    prev_layer->next_layer = self;
    self->next_layer = NULL;
    self->layer_id = prev_layer->layer_id + 1;

    self->depth = depth; // set all the parameters as entered by the user
    self->dimension = dimension;

    self->num_nodes = depth * dimension * dimension;

    self->output = (double *)calloc(self->num_nodes, sizeof(double)); // allocate memory for the output, gradient and errors arrays
    self->gradient = (double *)calloc(self->num_nodes, sizeof(double));
    self->errors = (double *)calloc(self->num_nodes, sizeof(double));

    self->num_biases = depth;
    // num weights = depth * prev_layer->depth * kernel_size * kernel_size
    self->num_weights = depth * prev_layer->depth * kernel_size * kernel_size;
    self->biases = (double *)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double *)calloc(self->num_biases, sizeof(double));

    self->weights = (double *)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double *)calloc(self->num_weights, sizeof(double));

    for (int i = 0; i < self->num_weights; i++)
        self->weights[i] = norm_rnd() * std;

    return self;
}

layer_component *create_full_layer(layer_component *prev_layer, int num_nodes, double std)
{
    // allocate memory for the layer
    layer_component *self = (layer_component *)calloc(1, sizeof(layer_component));

    self->prev_layer = prev_layer; // set the previous and next layers of prev_layer and self
    prev_layer->next_layer = self;
    self->next_layer = NULL;
    self->layer_id = prev_layer->layer_id + 1;

    self->type = FULL_LAYER; // set the type of the layer as FULL_LAYER

    self->depth = num_nodes;
    self->dimension = 1;

    self->num_nodes = self->depth; // set all the parameters as entered by the user

    self->output = (double *)calloc(self->num_nodes, sizeof(double));
    self->gradient = (double *)calloc(self->num_nodes, sizeof(double));
    self->errors = (double *)calloc(self->num_nodes, sizeof(double));

    self->num_biases = self->num_nodes;
    self->num_weights = num_nodes * prev_layer->num_nodes;

    self->biases = (double *)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double *)calloc(self->num_biases, sizeof(double));

    self->weights = (double *)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double *)calloc(self->num_weights, sizeof(double));

    for (int i = 0; i < self->num_weights; i++)
        // kaiming-He initialization
        self->weights[i] = norm_rnd() * std; // initialize the weights with random values

    return self;
}

void remove_layer(layer_component *self)
{ // free the memory allocated for the output, gradient, errors, biases and weights arrays
    free(self->output);
    free(self->gradient);
    free(self->errors);

    free(self->biases);
    free(self->weights);

    free(self->up_biases);
    free(self->up_weights);

    free(self);
}

void feedforwd_conv(layer_component *self)
{
    layer_component *prev_layer = self->prev_layer; // get the previous layer

    int kernsize = self->kernel_size; // get the kernel size
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) // iterate over the depth of the current layer
    {
        int qbase = z1 * prev_layer->depth * kernsize * kernsize; // qbase is the kernel matrix base index
        for (int y1 = 0; y1 < self->dimension; y1++)              // iterate over the dimension of the current layer
        {
            int y0 = self->stride * y1 - self->padding;  // y0 is the src pixel
            for (int x1 = 0; x1 < self->dimension; x1++) // iterate over the dimension of the current layer
            {
                int x0 = self->stride * x1 - self->padding; // for each pixel, get the src pixel
                double v = self->biases[z1];
                for (int z0 = 0; z0 < prev_layer->depth; z0++)
                {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * prev_layer->dimension * prev_layer->dimension; // pbase is the src matrix base index
                    for (int dy = 0; dy < kernsize; dy++)
                    {
                        int y = y0 + dy; // get the y value
                        if (0 <= y && y < prev_layer->dimension)
                        { // if y is within the bounds
                            int p = pbase + y * prev_layer->dimension;
                            int q = qbase + dy * kernsize;
                            for (int dx = 0; dx < kernsize; dx++) // iterate over the kernel size
                            {
                                int x = x0 + dx;
                                if (0 <= x && x < prev_layer->dimension)
                                {
                                    v += prev_layer->output[p + x] * self->weights[q + dx];
                                }
                            }
                        }
                    }
                }
                // apply the activation function
                v = relu(v);
                self->output[i] = v;
                self->gradient[i] = relu_grad(v);
                i++;
            }
        }
    }
}

void feedforwd_full(layer_component *self)
{ // get the previous layer
    layer_component *prev_layer = self->prev_layer;

    int k = 0;
    for (int i = 0; i < self->num_nodes; i++) // iterate over the number of nodes in the current layer
    {
        /* Compute Y = (W * X + B) without activation function. */
        double x = self->biases[i];
        for (int j = 0; j < prev_layer->num_nodes; j++)
        {
            x += (prev_layer->output[j] * self->weights[k++]);
        }
        self->output[i] = x;
    }

    if (self->next_layer == NULL) // softmax activation function for last layer
    {
        double m = -1; // softmax activation = exp(x-max(x))/sum(exp(x- max(x)))
        for (int i = 0; i < self->num_nodes; i++)
        {
            double x = self->output[i];
            if (m < x)
            {
                m = x;
            }
        }
        double t = 0;
        for (int i = 0; i < self->num_nodes; i++)
        {
            double x = self->output[i];
            double y = exp(x - m);
            self->output[i] = y;
            t += y;
        }
        for (int i = 0; i < self->num_nodes; i++)
        {
            self->output[i] /= t;
            self->gradient[i] = 1;
        }
    }
    else
    {
        // for all other layers, apply tanh activation function
        for (int i = 0; i < self->num_nodes; i++)
        {
            double x = self->output[i];
            double y = tanh(x);
            self->output[i] = y;
            self->gradient[i] = tanh_grad(y);
        }
    }
}

void set_input_layer(layer_component *self, const double *values)
{
    for (int i = 0; i < self->num_nodes; i++)
        self->output[i] = values[i]; // set output of input layer to the input values

    layer_component *cur_layer = self->next_layer; // get the next layer

    while (cur_layer != NULL)
    {
        if (cur_layer->kernel_size == 0 && cur_layer->stride == 0) // for full layer
            feedforwd_full(cur_layer);
        else
            feedforwd_conv(cur_layer); // for convolutional layer

        cur_layer = cur_layer->next_layer;
    }
}

void get_output(const layer_component *self, double *outputs)
{
    for (int i = 0; i < self->num_nodes; i++)
        outputs[i] = self->output[i]; // read the output values of the output layer
}

void feedback_full(layer_component *self)
{

    layer_component *lprev = self->prev_layer; // get the previous layer

    for (int j = 0; j < lprev->num_nodes; j++)
    {
        lprev->errors[j] = 0; // set the errors to 0 for the previous layer
    }

    int k = 0;
    for (int i = 0; i < self->num_nodes; i++)
    {
        double dnet = self->errors[i] * self->gradient[i]; // calculate the error = error * gradient
        for (int j = 0; j < lprev->num_nodes; j++)         // iterate over the number of nodes in the previous layer
        {
            lprev->errors[j] += self->weights[k] * dnet;    // calculate the error for the previous layer
            self->up_weights[k] += dnet * lprev->output[j]; // update the weights
            k++;
        }
        self->up_biases[i] += dnet; // update the biases
    }
}

void feedback_conv(layer_component *self)
{
    layer_component *lprev = self->prev_layer; //  get the previous layer

    for (int j = 0; j < lprev->num_nodes; j++) // error for the previous layer = 0
    {
        lprev->errors[j] = 0;
    }

    int kernsize = self->kernel_size;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) // iterate over the depth of the current layer
    {
        int qbase = z1 * lprev->depth * kernsize * kernsize; // qbase is the kernel matrix base index
        for (int y1 = 0; y1 < self->dimension; y1++)
        { // iterate over the dimension of the current layer
            int y0 = self->stride * y1 - self->padding;
            for (int x1 = 0; x1 < self->dimension; x1++)
            { // iterate over the dimension of the current layer for each getting pixel
                int x0 = self->stride * x1 - self->padding;
                double dnet = self->errors[i] * self->gradient[i];

                for (int dy = 0; dy < kernsize; dy++)
                { // iterate over the kernel size
                    int y = y0 + dy;
                    if (0 <= y && y < lprev->dimension)
                    { // if y is within the bounds
                        for (int dx = 0; dx < kernsize; dx++)
                        {
                            int x = x0 + dx;
                            if (0 <= x && x < lprev->dimension)
                            { // if x is within the bounds
                                lprev->errors[y * lprev->dimension + x] += self->weights[qbase + dy * kernsize + dx] * dnet;
                                // update the weights
                                if (self->gradient[i] > 0)
                                { // if gradient is greater than 0
                                    self->up_weights[qbase + dy * kernsize + dx] += dnet * lprev->output[y * lprev->dimension + x];
                                }
                            }
                        }
                    }
                }
                //  update the biases
                self->up_biases[z1] += dnet;
                i++;
            }
        }
    }
}

double get_total_error(layer_component *self)
{
    double total_error = 0;
    for (int i = 0; i < self->num_nodes; i++)
        total_error += self->errors[i] * self->errors[i]; // get average error over all nodes

    return (double)total_error / self->num_nodes;
}

void learn_output(layer_component *self, const double *values)
{
    for (int i = 0; i < self->num_nodes; i++)
    {
        self->errors[i] = (self->output[i] - values[i]);
    }
    /* Start backpropagation. */
    layer_component *layer = self;
    while (layer != NULL)
    { // iterate over all the layers
        switch (layer->type)
        {
        case FULL_LAYER:
            feedback_full(layer);
            break;
        case CONV_LAYER:
            feedback_conv(layer);
            break;
        case INPUT_LAYER:
            break;
        default:
            break;
        }
        layer = layer->prev_layer;
    }
}

void update_parameters(layer_component *self, double lr)
{

    for (int i = 0; i < self->num_biases; i++)
    {
        self->biases[i] -= lr * self->up_biases[i];
        self->up_biases[i] = 0; // update the biases
    }

    for (int i = 0; i < self->num_weights; i++)
    {
        self->weights[i] -= lr * self->up_weights[i];
        self->up_weights[i] = 0; // update the weights
    }

    if (self->prev_layer != NULL)
        update_parameters(self->prev_layer, lr); // for all the layers, update the parameters
}
