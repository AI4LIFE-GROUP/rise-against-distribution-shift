#!/bin/bash

python basic_rl_sepsis.py -e sepsis -a dp -ro -rt marginal -or cvar -rs 0.8 -sh 0.2 -n 100 -nr 1

# python basic_rl_sepsis.py -e sepsis -a dp -ro -rt marginal -or cvar -rs 0.8 -sh 0.2 -n 10000 -nr 10 &
# python basic_rl_sepsis.py -e sepsis -a dp -ro -rt joint -or cvar -rs 0.8 -sh 0.2 -n 10000 -nr 10

# echo "robust dp done rs=0.8"

# python basic_rl_sepsis.py -e sepsis -a dp -rt joint -sh 0.2 -n 10000 -nr 10

# echo "nominal dp done"
