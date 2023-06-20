!#/bin/sh
python main.py --mode describe --dataset ESSEX3IN1 --method switchCNN
python main.py --mode describe --dataset Nordlands --method switchCNN
python main.py --mode describe --dataset SPED_V2 --method switchCNN
python main.py --mode describe --dataset SFU --method switchCNN
python main.py --mode describe --dataset StLucia --method switchCNN



python main.py --mode eval_metrics --dataset SFU --method switchCNN
python main.py --mode eval_metrics --dataset GardensPointWalking --method switchCNN
python main.py --mode eval_metrics --dataset StLucia --method switchCNN
python main.py --mode eval_metrics --dataset Nordlands --method switchCNN
python main.py --mode eval_metrics --dataset ESSEX3IN1 --method switchCNN
python main.py --mode eval_metrics --dataset SPED_V2 --method switchCNN
