#include<stdint.h>

#ifndef MNIST_DATA_LOADER_H
#define MNIST_DATA_LOADER_H

typedef struct IdxFile{
    int num_dim;
    uint32_t *dims;
    uint8_t *data;
}IdxFile;

typedef struct header{
    uint16_t magic;
    uint8_t type;
    uint8_t num_dim;
}header;

uint32_t swap_endian(uint32_t value);
IdxFile* get_data(char path[]);
void delete_data(IdxFile* self);
uint8_t get_labels(IdxFile* self, int i);
void get_images(IdxFile* self, int i, uint8_t* output);

#endif
