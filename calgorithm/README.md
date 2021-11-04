## dependency installation

```
$ sudo apt install libeigen3-dev
```

## compile

```
$ g++ ica.cpp -I /usr/include/eigen3/ -fopenmp -lblas  
```
- `-O3 -mtune=native -march=native`は最適化系のオプション
- `-lblas`でblas利用（eigenのinclude前に`#define EIGEN_USE_BLAS`すること）
  - https://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html

## exec

```
$ OMP_NUM_THREADS=6 ./a.out
```