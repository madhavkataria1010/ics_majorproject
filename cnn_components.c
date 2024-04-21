#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "include/cnn_components.h"

double rnd(){
    double output = (double) rand()/RAND_MAX;
    return output;
}

double norm_rnd(){
    double output = (double) (rnd() + rnd() + rnd() + rnd() - 2)*1.724;
    return output;
}

double relu(double x){
    if (x>=0) return x;
    else return 0;
}

double relu_grad(double x){
    if (x>=0) return 1;
    else return 0;
}

double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2*x)) - 1.0;
}

double tanh_grad(double y)
{
    return (double) 1 - y*y;
}

layer_component* create_input_layer(int depth, int dimension){
    layer_component* self = (layer_component*) calloc(1, sizeof(layer_component));
    self->prev_layer = NULL;
    self->next_layer = NULL;
    self->layer_id = 0;

    self->depth = depth;
    self->dimension = dimension;
    self->type = INPUT_LAYER;

    self->num_nodes = depth*dimension*dimension;

    self->output = (double*) calloc(self->num_nodes, sizeof(double));
    self->gradient = (double*) calloc(self->num_nodes, sizeof(double));
    self->errors = (double*) calloc(self->num_nodes, sizeof(double));

    self->num_biases = 0;
    self->num_weights = 0;

    self->biases = (double*)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double*)calloc(self->num_biases, sizeof(double));

    self->weights = (double*)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double*)calloc(self->num_weights, sizeof(double));

    return self;
}

layer_component* create_conv_layer(layer_component* prev_layer, int depth, int dimension,
                                    int kernel_size, int padding, int stride, double std){

    layer_component* self = (layer_component*) calloc(1, sizeof(layer_component));
                 
    self->kernel_size = kernel_size;
    self->padding = padding;
    self->stride = stride;
    self->type = CONV_LAYER;

    self->prev_layer = prev_layer;
    prev_layer->next_layer = self;
    self->next_layer = NULL;
    self->layer_id = prev_layer->layer_id + 1;

    self->depth = depth;
    self->dimension = dimension;

    self->num_nodes = depth*dimension*dimension;

    self->output = (double*) calloc(self->num_nodes, sizeof(double));
    self->gradient = (double*) calloc(self->num_nodes, sizeof(double));
    self->errors = (double*) calloc(self->num_nodes, sizeof(double));

    self->num_biases = depth;
    self->num_weights = depth*prev_layer->depth*kernel_size*kernel_size;

    self->biases = (double*)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double*)calloc(self->num_biases, sizeof(double));

    self->weights = (double*)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double*)calloc(self->num_weights, sizeof(double));

    for (int i=0; i<self->num_weights; i++)
    self->weights[i] = norm_rnd()*std;

    return self;
}

layer_component* create_full_layer(layer_component* prev_layer, int num_nodes, double std){

    layer_component* self = (layer_component*) calloc(1, sizeof(layer_component));

    self->prev_layer = prev_layer;
    prev_layer->next_layer = self;
    self->next_layer = NULL;
    self->layer_id = prev_layer->layer_id + 1;

    self->type = FULL_LAYER;

    self->depth = num_nodes;
    self->dimension = 1;

    self->num_nodes = self->depth;

    self->output = (double*) calloc(self->num_nodes, sizeof(double));
    self->gradient = (double*) calloc(self->num_nodes, sizeof(double));
    self->errors = (double*) calloc(self->num_nodes, sizeof(double));

    self->num_biases = self->num_nodes;
    self->num_weights = num_nodes*prev_layer->num_nodes;

    self->biases = (double*)calloc(self->num_biases, sizeof(double));
    self->up_biases = (double*)calloc(self->num_biases, sizeof(double));

    self->weights = (double*)calloc(self->num_weights, sizeof(double));
    self->up_weights = (double*)calloc(self->num_weights, sizeof(double));

    for (int i=0; i<self->num_weights; i++)
    self->weights[i] = norm_rnd()*std;

    return self;
}

void remove_layer(layer_component* self){
    free(self->output);
    free(self->gradient);
    free(self->errors);

    free(self->biases);
    free(self->weights);

    free(self->up_biases);
    free(self->up_weights);

    free(self);
}

void feedforwd_conv(layer_component* self){
    layer_component* prev_layer = self->prev_layer;

    int kernsize = self->kernel_size;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * prev_layer->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->dimension; y1++) {
            int y0 = self->stride * y1 - self->padding;
            for (int x1 = 0; x1 < self->dimension; x1++) {
                int x0 = self->stride * x1 - self->padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double v = self->biases[z1];
                for (int z0 = 0; z0 < prev_layer->depth; z0++) {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * prev_layer->dimension * prev_layer->dimension;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < prev_layer->dimension) {
                            int p = pbase + y*prev_layer->dimension;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < prev_layer->dimension) {
                                    v += prev_layer->output[p+x] * self->weights[q+dx];
                                }
                            }
                        }
                    }
                }
                /* Apply the activation function. */
                v = relu(v);
                self->output[i] = v;
                self->gradient[i] = relu_grad(v);
                i++;
            }
        }
    }
}


void feedforwd_full(layer_component* self){
    layer_component* prev_layer = self->prev_layer;

    int k = 0;
    for (int i = 0; i < self->num_nodes; i++) {
        /* Compute Y = (W * X + B) without activation function. */
        double x = self->biases[i];
        for (int j = 0; j < prev_layer->num_nodes; j++) {
            x += (prev_layer->output[j] * self->weights[k++]);
        }
        self->output[i] = x;
    }

    if (self->next_layer == NULL) {
        /* Last layer - use Softmax. */
        double m = -1;
        for (int i = 0; i < self->num_nodes; i++) {
            double x = self->output[i];
            if (m < x) { m = x; }
        }
        double t = 0;
        for (int i = 0; i < self->num_nodes; i++) {
            double x = self->output[i];
            double y = exp(x-m);
            self->output[i] = y;
            t += y;
        }
        for (int i = 0; i < self->num_nodes; i++) {
            self->output[i] /= t;
            /* This isn't right, but set the same value to all the gradients. */
            self->gradient[i] = 1;
        }
    } else {
        /* Otherwise, use Tanh. */
        for (int i = 0; i < self->num_nodes; i++) {
            double x = self->output[i];
            double y = tanh(x);
            self->output[i] = y;
            self->gradient[i] = tanh_grad(y);
        }
    }
}


void set_input_layer(layer_component* self, const double* values){
    for (int i=0; i<self->num_nodes; i++)
    self->output[i] = values[i];

    layer_component* cur_layer = self->next_layer;

    while (cur_layer != NULL)
    {
        if (cur_layer->kernel_size == 0 && cur_layer->stride == 0) feedforwd_full(cur_layer);
        else feedforwd_conv(cur_layer);

        cur_layer = cur_layer->next_layer;
    }

    while (cur_layer != NULL) {
        switch (cur_layer->type) {
        case FULL_LAYER:
            feedforwd_full(cur_layer);
            break;
        case CONV_LAYER:
            feedforwd_conv(cur_layer);
            break;
        default:
            break;
        }
        cur_layer = cur_layer->next_layer;
    }
}

void get_output(const layer_component* self, double* outputs){
    for (int i=0; i<self->num_nodes; i++)
    outputs[i] = self->output[i];
}

void feedback_full(layer_component* self){
    
    layer_component* lprev = self->prev_layer;

    /* Clear errors. */
    for (int j = 0; j < lprev->num_nodes; j++) {
        lprev->errors[j] = 0;
    }

    int k = 0;
    for (int i = 0; i < self->num_nodes; i++) {
        /* Computer the weight/bias updates. */
        double dnet = self->errors[i] * self->gradient[i];
        for (int j = 0; j < lprev->num_nodes; j++) {
            /* Propagate the errors to the previous layer. */
            lprev->errors[j] += self->weights[k] * dnet;
            self->up_weights[k] += dnet * lprev->output[j];
            k++;
        }
        self->up_biases[i] += dnet;
    }
}

void feedback_conv(layer_component* self) {
  layer_component* lprev = self->prev_layer;

  /* Clear errors. */
  for (int j = 0; j < lprev->num_nodes; j++) {
    lprev->errors[j] = 0;
  }

  int kernsize = self->kernel_size;
  int i = 0;
  for (int z1 = 0; z1 < self->depth; z1++) {
    /* z1: dst matrix */
    /* qbase: kernel matrix base index */
    int qbase = z1 * lprev->depth * kernsize * kernsize;
    for (int y1 = 0; y1 < self->dimension; y1++) {
      int y0 = self->stride * y1 - self->padding;
      for (int x1 = 0; x1 < self->dimension; x1++) {
        int x0 = self->stride * x1 - self->padding;
        double dnet = self->errors[i] * self->gradient[i];

        for (int dy = 0; dy < kernsize; dy++) {
          int y = y0 + dy;
          if (0 <= y && y < lprev->dimension) {
            for (int dx = 0; dx < kernsize; dx++) {
              int x = x0 + dx;
              if (0 <= x && x < lprev->dimension) {
                /* 
                 * Calculate dnet considering the activation gradient 
                 * of the output neuron (self->gradient[i])
                 */
                

                /* Propagate the error to the previous layer */
                lprev->errors[y * lprev->dimension + x] += self->weights[qbase + dy * kernsize + dx] * dnet;

                /* 
                 * Update weight only if the corresponding output neuron 
                 * in the previous layer was active (gradient > 0)
                 */
                if (self->gradient[i] > 0) {
                  self->up_weights[qbase + dy * kernsize + dx] += dnet * lprev->output[y * lprev->dimension + x];
                }
              }
            }
          }
        }
        self->up_biases[z1] += dnet;
        i++;
      }
    }
  }
}


double get_total_error(layer_component* self){
    double total_error = 0;
    for (int i=0; i<self->num_nodes; i++)
    total_error += self->errors[i]*self->errors[i];

    return (double) total_error/self->num_nodes;
}

void learn_output(layer_component* self, const double* values)
{
    for (int i = 0; i < self->num_nodes; i++) {
        self->errors[i] = (self->output[i] - values[i]);
    }
    /* Start backpropagation. */
    layer_component* layer = self;
    while (layer != NULL) {
        switch (layer->type) {
        case FULL_LAYER:
            feedback_full(layer);
            break;
        case CONV_LAYER:
            feedback_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->prev_layer;
    }
}


void update_parameters(layer_component* self, double lr){

    for (int i=0; i<self->num_biases; i++){
        self->biases[i] -= lr*self->up_biases[i];
        self->up_biases[i] = 0;
    }

    for (int i=0; i<self->num_weights; i++){
        self->weights[i] -= lr*self->up_weights[i];
        self->up_weights[i] = 0;
    }

    if (self->prev_layer != NULL)
    update_parameters(self->prev_layer, lr);
}

// void update_parameters(layer_component* self, double lr) {
//     // Update biases
//     if (self->num_biases > 0) {
//         double bias_update = 0.0;
//         for (int i = 0; i < self->num_biases; i++) {
//             bias_update += self->up_biases[i];
//         }
//         bias_update /= self->num_biases; // Average over the entire batch
//         for (int i = 0; i < self->num_biases; i++) {
//             self->biases[i] -= lr * bias_update;
//         }
//     }

//     // Update weights
//     for (int i = 0; i < self->num_weights; i++) {
//         self->weights[i] -= lr * self->up_weights[i];
//     }

//     // Reset gradients
//     memset(self->up_biases, 0, self->num_biases * sizeof(double));
//     memset(self->up_weights, 0, self->num_weights * sizeof(double));

//     // Recursively update parameters for previous layers
//     if (self->prev_layer != NULL) {
//         update_parameters(self->prev_layer, lr);
//     }
// }
