#!/bin/bash
for i in $(seq 0 0.1 1);
do
  python3 test_nms.py $i;
done
