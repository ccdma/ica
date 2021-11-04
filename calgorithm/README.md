## dependency installation

```
$ sudo apt install libeigen3-dev
```

## compile

```
$ g++ ica.cpp -I /usr/include/eigen3/ -fopenmp
```


## exec

```
$ OMP_NUM_THREADS=6 ./a.out
```