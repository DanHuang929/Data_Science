## python version
Python 3.9.7

## 用到的package
attacker.py:
import torch (用來計算gradient)
from core import utils (將array轉成tensor、做normalization)

main.py
import pickle (load data)
from pathlib import Path (load data)

## 執行指令
python main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt --use_gpu