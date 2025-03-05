# This is the code of CognTKE

### Process data
First, unpack the data files data.zip

### Train models

Then, the following commands can be used to train the proposed models. 

'''

python  main.py  --n_layer 3 --window_size 15 --gpu 1 --dataset ICEWS14  --model_name TRED_GNN  --batch_size 128 --train_mode half

'''
