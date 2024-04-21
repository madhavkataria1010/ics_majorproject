#include "cnn_components.h"
#ifndef MODEL_SAVER_H
#define MODEL_SAVER_H

void save_model(layer_component* self);

void load_model(layer_component* self);

void init_model_architecture(layer_component** linput, layer_component** lconv1, layer_component** lconv2,
                                layer_component** lfull1, layer_component** lfull2, layer_component** loutput);

#endif
