all:
	gcc -o test test_model.c cnn_components.c MNIST_data_loader.c model_saver.c -lm
	gcc -o main train_model.c cnn_components.c MNIST_data_loader.c model_saver.c -lm
