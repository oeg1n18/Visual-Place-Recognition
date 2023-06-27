!#/bin/sh



python main.py --mode eval_metrics --dataset SFU --method switchCNN_f1
python main.py --mode eval_metrics --dataset GardensPointWalking --method switchCNN_f1
python main.py --mode eval_metrics --dataset StLucia --method switchCNN_f1
python main.py --mode eval_metrics --dataset Nordlands --method switchCNN_f1
python main.py --mode eval_metrics --dataset ESSEX3IN1 --method switchCNN_f1
python main.py --mode eval_metrics --dataset SPED_V2 --method switchCNN_f1



python main.py --mode eval_metrics --dataset SFU --method switchCNN_prec
python main.py --mode eval_metrics --dataset GardensPointWalking --method switchCNN_prec
python main.py --mode eval_metrics --dataset StLucia --method switchCNN_prec
python main.py --mode eval_metrics --dataset Nordlands --method switchCNN_prec
python main.py --mode eval_metrics --dataset ESSEX3IN1 --method switchCNN_prec
python main.py --mode eval_metrics --dataset SPED_V2 --method switchCNN_prec
