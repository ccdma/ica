## dependency installation

```
$ sudo apt install libeigen3-dev
```

## compile

```
$ g++ ica.cpp -I /usr/include/eigen3/ -fopenmp -lblas -O3 -mtune=native -march=native 
```
`blas`以降は最適化系のオプション

## exec

```
$ OMP_NUM_THREADS=6 ./a.out
```