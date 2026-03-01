#!/bin/bash
cppcheck --enable=all --inconclusive \
         -i json.hpp \
         serial.cpp



