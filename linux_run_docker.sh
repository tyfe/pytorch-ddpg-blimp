#!/bin/bash
# GNU bash, version 4.3.48(1)-release(x86_64-pc-linux-gnu) 
docker run -it \
-v /home/hri/Documents/RL_Book/:/tf/rl \
-p 8888:8888 -p 5005:5005 rl_book_tensorflow /bin/bash
