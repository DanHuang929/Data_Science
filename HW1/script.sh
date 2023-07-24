#!/bin/bash
g++ -std=c++2a -pthread -O2 -o 108062313_hw1 108062313_hw1.cpp
./108062313_hw1 0.2 sample.txt test_out1.txt
diff sample_out.txt test_out1.txt 