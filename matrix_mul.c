#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

typedef struct _Matrix {
    int **v;
    int size;
    bool shadow_copy;
} Matrix;

typedef struct _ThreadParam {
    Matrix *A, *B, *M, *tmp;
    int id;
} ThreadParam;

#define STRASSEN_THRESHOLD 1
bool TRANSPOSE = false;
bool SHADOW_COPY = false;
bool KEEP_STRASSEN = false;

void strassen_mul(const Matrix *A_all, const Matrix *B_all, Matrix *C_all, bool parallel);

void matrix_free(Matrix *thiz)
{
    assert(thiz != NULL);
    assert(thiz->v != NULL);
    if (!thiz->shadow_copy) {
        free(thiz->v[0]);
    }
    free(thiz->v);
    thiz->v = NULL;
}

void matrixs_free(Matrix *thiz, int num)
{
    for (int i = 0; i < num; ++i)
        matrix_free(&thiz[i]);
}

void matrix_create(Matrix *thiz, int size)
{
    thiz->size = size;

    // dynamic allocate 2d matrix
    int *arr = (int*)malloc(size * size * sizeof(int));
    thiz->v = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i)
        thiz->v[i] = arr + i*size;
}

void matrix_try_create(Matrix *thiz, int size)
{
    assert(thiz != NULL);
    if (thiz->size == 0)
        matrix_create(thiz, size);
}

void matrix_read(Matrix *thiz, FILE *fp)
{
    assert(fp != NULL);
    int m, n;
    if (fscanf(fp, "%d %d", &m, &n) != 2) {
        perror("when read size");
        exit(EXIT_FAILURE);
    }
    assert(m == n);
    matrix_try_create(thiz, m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (fscanf(fp, "%d", &(thiz->v[i][j])) != 1) {
                perror("when read size");
                exit(EXIT_FAILURE);
            }
        }
    }
}

void matrix_print(Matrix thiz)
{
    for (int i = 0; i < thiz.size; ++i) {
        for (int j = 0; j < thiz.size; ++j)
            printf("%d ", thiz.v[i][j]);
        puts("");
    }
    puts("");
}

void matrix_check_AB_try_create_C(const Matrix *A, const Matrix *B, Matrix *C)
{
    assert(A != NULL && B != NULL);
    assert(A->size == B->size);
    matrix_try_create(C, A->size);
}

void matrix_add(const Matrix *A, const Matrix *B, Matrix *C)
{
    matrix_check_AB_try_create_C(A, B, C);
    int size = A->size;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C->v[i][j] = A->v[i][j] + B->v[i][j];
        }
    }
}

void matrix_sub(const Matrix *A, const Matrix *B, Matrix *C)
{
    matrix_check_AB_try_create_C(A, B, C);
    int size = A->size;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C->v[i][j] = A->v[i][j] - B->v[i][j];
        }
    }
}

void matrix_mul(const Matrix *A, const Matrix *B, Matrix *C)
{
    matrix_check_AB_try_create_C(A, B, C);
    int size = A->size;
    for (int i = 0; i < size; ++i)
        memset(C->v[i], 0, size * sizeof(int));

    if(TRANSPOSE) {
        for (int i = 0; i < size; ++i) {
            for (int k = 0; k < size; ++k) {
                for (int j = 0; j < size; ++j) {
                    C->v[i][j] += A->v[i][k] * B->v[k][j];
                }
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    C->v[i][j] += A->v[i][k] * B->v[k][j];
                }
            }
        }
    }
}

void matrix_divide_4(const Matrix *thiz, Matrix *block)
{
    int size = thiz->size;
    int size_divided = size/2;
    for (int i = 0; i < 4; ++i)
        matrix_try_create(&block[i], size_divided);
    if (SHADOW_COPY) {
        for (int i = 0; i < 4; ++i) {
            free(block[i].v[0]);
            block[i].shadow_copy = true;
        }
        for (int i = 0; i < size_divided; ++i) {
            block[0].v[i] = thiz->v[i];
            block[1].v[i] = thiz->v[i] + size_divided;
            block[2].v[i] = thiz->v[i+size_divided];
            block[3].v[i] = thiz->v[i+size_divided] + size_divided;
        }
    } else {
        for (int i = 0; i < size_divided; ++i) {
            for (int j = 0; j < size_divided; ++j) {
                block[0].v[i][j] = thiz->v[i][j];
                block[1].v[i][j] = thiz->v[i][j+size_divided];
                block[2].v[i][j] = thiz->v[i+size_divided][j];
                block[3].v[i][j] = thiz->v[i+size_divided][j+size_divided]; 
            }
        }
    }
}

void matrix_combine_4(Matrix *thiz, const Matrix *block)
{
    int size_divided = block[0].size;
    int size= 2*size_divided;
    matrix_try_create(thiz, size);
    for (int i = 0; i < size_divided; ++i) {
        for (int j = 0; j < size_divided; ++j) {
            thiz->v[i][j] = block[0].v[i][j];
            thiz->v[i][j+size_divided] = block[1].v[i][j];
            thiz->v[i+size_divided][j] = block[2].v[i][j];
            thiz->v[i+size_divided][j+size_divided] = block[3].v[i][j];
        }
    }
}

void strassen_mul_no_parallel(const Matrix *A, const Matrix *B, Matrix *C)
{
    strassen_mul(A, B, C, false);
}

void _strassen_mul(const Matrix *A, const Matrix *B, Matrix *M, Matrix *tmp, const int id)
{
    void (*mul_func)(const Matrix*, const Matrix*, Matrix*) = matrix_mul;
    if (KEEP_STRASSEN && A->size > STRASSEN_THRESHOLD) {
        mul_func = strassen_mul_no_parallel;
    }

    if (id == -1 || id == 0) {
        matrix_add(&A[0], &A[3], &tmp[0]);
        matrix_add(&B[0], &B[3], &tmp[1]);
        mul_func(&tmp[0], &tmp[1], &M[0]);
    }
    if (id == -1 || id == 1) {
        matrix_add(&A[2], &A[3], &tmp[2]);
        mul_func(&tmp[2], &B[0], &M[1]);
    }
    if (id == -1 || id == 2) {
        matrix_sub(&B[1], &B[3], &tmp[3]);
        mul_func(&A[0], &tmp[3], &M[2]);
    }
    if (id == -1 || id == 3) {
        matrix_sub(&B[2], &B[0], &tmp[4]);
        mul_func(&A[3], &tmp[4], &M[3]);
    }
    if (id == -1 || id == 4) {
        matrix_add(&A[0], &A[1], &tmp[5]);
        mul_func(&tmp[5], &B[3], &M[4]);
    }
    if (id == -1 || id == 5) {
        matrix_sub(&A[2], &A[0], &tmp[6]);
        matrix_add(&B[0], &B[1], &tmp[7]);
        mul_func(&tmp[6], &tmp[7], &M[5]);
    }
    if (id == -1 || id == 6) {
        matrix_sub(&A[1], &A[3], &tmp[8]);
        matrix_add(&B[2], &B[3], &tmp[9]);
        mul_func(&tmp[8], &tmp[9], &M[6]);
    }
}

void* pthread_func(void *arg)
{
    ThreadParam *t = (ThreadParam*)arg;
    _strassen_mul(t->A, t->B, t->M, t->tmp, t->id);
    pthread_exit(NULL);
}

void strassen_mul(const Matrix *A_all, const Matrix *B_all, Matrix *C_all, bool parallel)
{
    Matrix A[4] = {0},
           B[4] = {0},
           C[4] = {0},
           M[7] = {0},
           tmp[10] = {0};

    matrix_divide_4(A_all, A);
    matrix_divide_4(B_all, B);

    if (parallel) {
        int num_thread = 7;
        pthread_t thread[num_thread];
        ThreadParam thread_param[num_thread];

        for(int i = 0; i < num_thread; i++) {
            ThreadParam t = {.A=A, .B=B, .M=M, .tmp=tmp, .id = i};
            memcpy(&thread_param[i], &t, sizeof(ThreadParam));
            pthread_create(&thread[i], NULL, pthread_func, &thread_param[i]);
        }
        for(int i = 0; i < num_thread; i++) {
            pthread_join(thread[i], NULL);
        }
    } else {
        _strassen_mul(A, B, M, tmp, -1);
    }

    matrix_add(&M[0], &M[3], &tmp[0]);
    matrix_sub(&tmp[0], &M[4], &tmp[0]);
    matrix_add(&tmp[0], &M[6], &C[0]);
    matrix_add(&M[2], &M[4], &C[1]);
    matrix_add(&M[1], &M[3], &C[2]);
    matrix_sub(&M[0], &M[1], &tmp[0]);
    matrix_add(&tmp[0], &M[2], &tmp[0]);
    matrix_add(&tmp[0], &M[5], &C[3]);

    matrix_combine_4(C_all, C);

    matrixs_free(A, 4);
    matrixs_free(B, 4);
    matrixs_free(C, 4);
    matrixs_free(M, 7);
    matrixs_free(tmp, 10);
}

int main(int argc, const char **argv)
{
    if (argc < 3) {
        puts("./main path/to/test/data type_of_mul [print? (0,1)]");
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen(argv[1], "r");
    if (fp==NULL) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }
    int mul_type = atoi(argv[2]);

    Matrix A = {0},
           B = {0},
           C = {0};
    matrix_read(&A, fp);
    matrix_read(&B, fp);
    fclose(fp);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    switch (mul_type) {
        case 0:
            puts("ordinary");
            matrix_mul(&A, &B, &C);
            break;
        case 1:
            puts("ordinary + cache friendly");
            TRANSPOSE = true;
            matrix_mul(&A, &B, &C);
            break;
        case 2:
            puts("strassen + cache friendly");
            TRANSPOSE = true;
            strassen_mul(&A, &B, &C, false);
            break;
        case 3:
            puts("strassen + cache friendly + multithread");
            TRANSPOSE = true;
            strassen_mul(&A, &B, &C, true);
            break;
        case 4:
            puts("strassen + cache friendly + multithread + keep strassen");
            TRANSPOSE = true;
            KEEP_STRASSEN = true;
            strassen_mul(&A, &B, &C, true);
            break;
        case 5:
            puts("strassen + cache friendly + multithread + keep strassen + shadow copy");
            TRANSPOSE = true;
            KEEP_STRASSEN = true;
            SHADOW_COPY = true;
            strassen_mul(&A, &B, &C, true);
            break;
        case 6:
            puts("strassen + multithread + keep strassen + shadow copy");
            KEEP_STRASSEN = true;
            SHADOW_COPY = true;
            strassen_mul(&A, &B, &C, true);
            break;
        default:
            fprintf(stderr, "wrong mul type (0~6)");
            exit(EXIT_FAILURE);
    }

    if (argc == 4 && atoi(argv[3]) == true)
        matrix_print(C);

    gettimeofday(&end, NULL);
    int time_spent = (1e6) * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
    printf("%d ms\n",time_spent/1000);

    return 0;
}
