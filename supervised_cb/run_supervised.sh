#!/bin/bash

python full_med_experiments.py --filename medce --alpha 0.2 --lip 0.1 --dlr 0.01 --ntrain 100 --ntest 100 --steps 100 --kdual 2 --trshift 0.8


# ntrain=2000
# ntest=2000
# steps=10000

# for alpha in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}
# do
#     for lip in {0.01,}
#     do
#         for dlr in {0.001,0.01,0.1,}
#         do
#             echo "$ntrain,$ntest,$alpha,$lip,$dlr,$steps"
#             python full_med_experiments.py --filename medce --alpha "$alpha" --lip "$lip" --dlr "$dlr" --ntrain "$ntrain" --ntest "$ntest" --steps "$steps" --kdual 2 --trshift 0.8 &
#         done
#     done
#     wait
#     echo "done $alpha"
# done