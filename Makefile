CC=gcc
CFLAGS=-I.

DEPS = src/cnn_components.h src/MNIST_data_loader.h src/model_saver.h

OBJ1 = src/test_model.o src/cnn_components.o src/MNIST_data_loader.o src/model_saver.o
OBJ2 = src/train_model.o src/cnn_components.o src/MNIST_data_loader.o src/model_saver.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
	

all: test main
	rm -f src/*.o


test: $(OBJ1)
	$(CC) -o $@ $^ $(CFLAGS) -lm

main: $(OBJ2)
	$(CC) -o $@ $^ $(CFLAGS) -lm

clean:
	rm -f src/*.o test main

