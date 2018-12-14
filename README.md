# Programming Language Homework 4 報告
## 傳統 Matrix Multiplication 與 Strassen's Matrix Multiplication 平行運算

[GitHub siahuat0727/matrix-multiplication](https://github.com/siahuat0727/matrix-multiplication)

## 語言

C

## 編譯

正常編譯
```
make
```
最佳化編譯
```
make opt
```

使用
```
./main path/to/test/data mul_method [print_result]
```
mul_method 對應以下的 0~6 個方法
print_result 爲 1 時將輸出結果

例子
```
./main test3 5
./main test2 4 1 > ans.txt
```

## 方法
### 0. Ordinary
傳統矩陣相乘
```c
for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
            C->v[i][j] += A->v[i][k] * B->v[k][j];
        }
    }
}
```
### 1. Ordinary + Cache friendly
考慮 cache 存取的情況調整拜訪矩陣元素的順序
```c
for (int i = 0; i < size; ++i) {
    for (int k = 0; k < size; ++k) {
        for (int j = 0; j < size; ++j) {
            C->v[i][j] += A->v[i][k] * B->v[k][j];
        }
    }
}
```
### 2. Strassen + Cache friendly
略

### 3. Strassen + Cache friendly + Multithread
使用 pthread 實作，分出7個 thread，每個 thread 執行 七個 strassen 小矩陣(4分之1)乘法的其中一個

### 4. Strassen + Cache friendly + Multithread + Keep Strassen
方法三只會在第一次相乘時使用 Strassen 來切，而接下來用傳統的乘法去成，但其實接下來也可以繼續用 Strassen 切，當然切太小反而會變慢。
我們經過實驗發現切到 32×32 就不再切會是最快的。
```c
void strassen_mul(const Matrix *A_all, const Matrix *B_all, Matrix *C_all, bool parallel)
{    
    int size = A_all->size;
    if (size <= STRASSEN_THRESHOLD) { // STRASSEN_THRESHOLD = 32
        matrix_mul(A_all, B_all, C_all);
        return;
    }
    ...
}
```
### 5. Strassen + Cache friendly + Multithread + Keep Strassen + Shallow copy
Strassen 中需要把矩陣切成四小塊，那比較直覺的做法就會是直接 copy 四個小矩陣出來，而我們嘗試讓小 matrix 用 pointer 指向大 matrix 對應的位置，這樣就可以節省 copy 的時間和存值的空間了。

![](https://i.imgur.com/UX4KXhv.png)

詳見 code
```c
typedef struct _Matrix {
    int **v; 
    int size;
    bool shallow_copy;
} Matrix; 
```
```c
void matrix_divide_4(const Matrix *thiz, Matrix *blocks)
{            
    int size = thiz->size;
    int size_divided = size/2;
    for (int i = 0; i < 4; ++i)
        matrix_try_create(&blocks[i], size_divided);
    if (SHALLOW_COPY) { // shallow copy
        for (int i = 0; i < 4; ++i) {
            free(blocks[i].v[0]);
            blocks[i].shallow_copy = true;
        }    
        for (int i = 0; i < size_divided; ++i) {
            blocks[0].v[i] = thiz->v[i];
            blocks[1].v[i] = thiz->v[i] + size_divided;
            blocks[2].v[i] = thiz->v[i+size_divided];
            blocks[3].v[i] = thiz->v[i+size_divided] + size_divided;
        }    
    } else { // 逐個 element copy 值
        for (int i = 0; i < size_divided; ++i) {
            for (int j = 0; j < size_divided; ++j) {                                                    
                blocks[0].v[i][j] = thiz->v[i][j];
                blocks[1].v[i][j] = thiz->v[i][j+size_divided];
                blocks[2].v[i][j] = thiz->v[i+size_divided][j];
                blocks[3].v[i][j] = thiz->v[i+size_divided][j+size_divided];
            }
        }    
    }        
}       
```

#### Deep copy
Time complexity: $O(nm)$
Space complexity: $O(nm)$

#### Shallow copy
Time complexity: $O(n)$
Space complexity: $O(n)$

$n$: number of row of matrix
$m$: number of column of matrix


### 6. Strassen + Multithread + Keep Strassen + Shallow copy
與方法 5 的差別是少了 cache friendly。
這是因爲後來發現用了 Keep Strassen 之後，大小爲 1024 和 4096 的測資在沒 Cache friendly 的情況下會跑得比較好，其原因還有待思考，但在加了 -O3 flag 之後就不會有這不正常的現象。（沒 cache friendly 終於比較慢了）

### 7. 外掛 optimization flag
優化怎麼能忘了 GNU 自帶的 optimization flag！
以上的最好方法計算 4096×4096 需要 30++ 秒，使用 gcc 的 -O3 optimization flag 之後只需 3.++ 秒！

## 效能分析

### 1024 * 1024

![](https://i.imgur.com/MnCcinJ.png)

### 4096 * 4096

![](https://i.imgur.com/MzZKOH5.png)

![](https://i.imgur.com/MvkdO4a.png)

### 4096 * 4096 with optimization flag -O3

![](https://i.imgur.com/CVZJthF.png)

![](https://i.imgur.com/NpISlPs.png)

