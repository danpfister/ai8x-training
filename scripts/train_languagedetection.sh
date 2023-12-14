#!/bin/sh
python train.py --epochs 30 --optimizer Adam --lr 0.001 --deterministic --model ai85languagenet --use-bias --param-hist --batch-size 256 --dataset LanguageDetection --confusion --device MAX78000 "$@"
