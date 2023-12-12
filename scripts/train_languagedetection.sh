#!/bin/sh
python train.py --epochs 10 --optimizer Adam --lr 0.001 --wd 0 --deterministic --model ai85languagenet --use-bias --dataset LanguageDetection --confusion --device MAX78000 "$@"
