all:
	gcc -Wall -o main matrix_mul.c -lpthread

opt:
	gcc -Wall -O3 -o main matrix_mul.c -lpthread

clean:
	rm main
