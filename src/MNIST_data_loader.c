#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/MNIST_data_loader.h"

uint32_t swap_endian(uint32_t value) {
    return ((value >> 24) & 0xff) |     // move byte 3 to byte 0
           ((value << 8) & 0xff0000) |  // move byte 1 to byte 2
           ((value >> 8) & 0xff00) |    // move byte 2 to byte 1
           ((value << 24) & 0xff000000);// byte 0 to byte 3
}

IdxFile* get_data(char path[]){
    FILE* fp = fopen(path, "rb");
    
    if (fp == NULL) return NULL;

    struct header header;
    fread(&header, sizeof(header), 1, fp);

    if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.num_dim < 1) return NULL;

    IdxFile* self = (IdxFile*) calloc(1, sizeof(IdxFile));

    self->num_dim = header.num_dim;
    self->dims = (uint32_t*)calloc(self->num_dim, sizeof(uint32_t));
    fread(self->dims, sizeof(uint32_t), self->num_dim, fp);

    uint32_t total_bytes = sizeof(uint8_t);
    for (int i=0; i<self->num_dim; i++){
        uint32_t size = swap_endian(self->dims[i]);
        self->dims[i] = size;
        total_bytes *= size;
    }

    self->data = (uint8_t*) malloc(total_bytes);
    fread(self->data, sizeof(uint8_t), total_bytes, fp);
    fclose(fp);
    return self;
}

void delete_data(IdxFile* self){
    free(self->dims);
    free(self->data);
    self->dims = NULL;
    self->data = NULL;
    free (self);
}

uint8_t get_labels(IdxFile* self, int i){
    return self->data[i];
}

void get_images(IdxFile* self, int i, uint8_t* output){
    size_t size = self->dims[1] * self->dims[2];
    memcpy(output, &self->data[i*size], size);
}
