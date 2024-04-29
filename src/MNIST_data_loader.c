#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/MNIST_data_loader.h"

// Function to swap the endianess of a 32-bit integer
uint32_t swap_endian(uint32_t value) {
    return ((value >> 24) & 0xff) |     // move byte 3 to byte 0
           ((value << 8) & 0xff0000) |  // move byte 1 to byte 2
           ((value >> 8) & 0xff00) |    // move byte 2 to byte 1
           ((value << 24) & 0xff000000);// byte 0 to byte 3
}

IdxFile* get_data(char path[]){
    FILE* fp = fopen(path, "rb");
    
    // Check if file is opened
    if (fp == NULL) return NULL;
    // Read the header
    struct header header;
    fread(&header, sizeof(header), 1, fp);
    // Check if the file is IDX
    if (header.magic != 0) return NULL;
    // Check if the file is a label file
    if (header.type != 0x08) return NULL;
    // Check if the file has at least one dimension
    if (header.num_dim < 1) return NULL;
    // Check if the file has at least one data point
    IdxFile* self = (IdxFile*) calloc(1, sizeof(IdxFile));
    // Read the dimensions
    self->num_dim = header.num_dim;
    // Allocate memory for the dimensions
    self->dims = (uint32_t*)calloc(self->num_dim, sizeof(uint32_t));
    // Read the dimensions
    fread(self->dims, sizeof(uint32_t), self->num_dim, fp);
    // Calculate the total number of bytes  

    uint32_t total_bytes = sizeof(uint8_t);
    for (int i=0; i<self->num_dim; i++){
        uint32_t size = swap_endian(self->dims[i]);
        self->dims[i] = size;
        total_bytes *= size;
    }

//    printf("Total bytes: %d\n", total_bytes);
    self->data = (uint8_t*) malloc(total_bytes);
    // Read the data
    fread(self->data, sizeof(uint8_t), total_bytes, fp);
    fclose(fp);
    return self;
}

void delete_data(IdxFile* self){
    // Free the memory
    free(self->dims);
    free(self->data);
    self->dims = NULL;
    self->data = NULL;
    free (self);
}

// Getters
uint8_t get_labels(IdxFile* self, int i){
    // Check if the index is valid
    return self->data[i];
}

void get_images(IdxFile* self, int i, uint8_t* output){
    // Check if the index is valid
    size_t size = self->dims[1] * self->dims[2];
    // Copy the image
    memcpy(output, &self->data[i*size], size);
}
