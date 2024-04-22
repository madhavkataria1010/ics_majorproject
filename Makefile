all:
	gcc -o test src/test_model.c src/cnn_components.c src/MNIST_data_loader.c src/model_saver.c -lm
	gcc -o main src/train_model.c src/cnn_components.c src/MNIST_data_loader.c src/model_saver.c -lm
