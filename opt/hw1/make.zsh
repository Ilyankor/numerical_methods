#!/bin/zsh

cd src
gfortran -O3 -llapack -march=native -c hw1.f90 -o hw1_f.o
g++ -std=c++23 -c hw1.cpp -o hw1_c.o
g++ -llapack -lgfortran hw1_c.o hw1_f.o -o ../main
rm -f hw1_f.o hw1_c.o
cd ../
