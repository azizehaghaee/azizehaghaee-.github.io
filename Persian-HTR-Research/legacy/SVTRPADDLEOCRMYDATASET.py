# Historical PaddleOCR/SVTR experiment preserved from the project library.
# This file contains notebook magics and is intended to be run as a notebook cell.

!pip install paddlepaddle-gpu==2.6.1 -q
!pip install paddleocr -q
!pip install shapely pyclipper lmdb tqdm -q

!git clone https://github.com/PaddlePaddle/PaddleOCR.git
%cd PaddleOCR

import paddle
print("Paddle:", paddle.__version__)
print("CUDA:", paddle.device.is_compiled_with_cuda())
print("Device:", paddle.device.get_device())

from google.colab import drive
drive.mount('/content/drive')

import os, random
DATA_DIR = "/home/root1/projects/htr/my_dataset_final/labeled"
OUTPUT_DIR = "/home/root1/projects/htr/output_model"
INFERENCE_DIR = "/home/root1/projects/htr/inference_images"
INFERENCE_CSV = "/home/root1/projects/htr/inference_output.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(INFERENCE_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR,"labels.txt"),encoding="utf8") as f: lines=f.readlines()
random.shuffle(lines); split=int(len(lines)*0.9); train_lines=lines[:split]; val_lines=lines[split:]
open("train.txt","w",encoding="utf8").writelines(train_lines); open("val.txt","w",encoding="utf8").writelines(val_lines)
print(len(train_lines),len(val_lines))

chars=set()
for line in lines:
    txt=line.strip().split("|")[1]; chars.update(list(txt))
chars=sorted(chars)
with open("persian_dict.txt","w",encoding="utf8") as f:
    for c in chars: f.write(c+"\n")
print("chars =",len(chars))

!cp configs/rec/rec_svtrnet.yml svtr_persian.yml
import yaml
cfg=yaml.safe_load(open("svtr_persian.yml"))
cfg["Global"]["character_dict_path"]="persian_dict.txt"
cfg["Train"]["dataset"]["data_dir"]=DATA_DIR
cfg["Train"]["dataset"]["label_file_list"]=["train.txt"]
cfg["Eval"]["dataset"]["data_dir"]=DATA_DIR
cfg["Eval"]["dataset"]["label_file_list"]=["val.txt"]
cfg["Global"]["epoch_num"]=100
cfg["Global"]["save_model_dir"]="./svtr_persian"
cfg["Optimizer"]["lr"]["learning_rate"]=0.0005
cfg["Architecture"]["algorithm"]="SVTR"
cfg["Train"]["loader"]["batch_size_per_card"]=32
cfg["Eval"]["loader"]["batch_size_per_card"]=32
with open("svtr_persian.yml","w") as f: yaml.dump(cfg,f)

!pip install rapidfuzz -q
!python tools/train.py -c svtr_persian.yml
!python tools/eval.py -c svtr_persian.yml -o Global.pretrained_model=./svtr_persian/best_accuracy
!python tools/infer_rec.py -c svtr_persian.yml -o Global.pretrained_model=./svtr_persian/best_accuracy Global.infer_img=/content/sample.tif

!pip install jiwer
from jiwer import cer, wer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(7,5))
models=["CRNN","ViTSTR","SVTR-Large"]
# Historical source ended with: plt.bar(models, cers)
