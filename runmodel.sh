#!/bin/bash

docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py
