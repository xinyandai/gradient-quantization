#  quantization for stochastic gradient
## quantization methods
* sgd
* qsgd
* hsq
* nnq
## implementation 
Run

    python main.py  --quantizer hsq --num-users 8 --batch-size 64 --save-log logs/hsq_u8_b64
    python main.py  --quantizer nnq --num-users 8 --batch-size 64 --save-log logs/nnq_u8_b64
    python main.py  --quantizer sgd --num-users 8 --batch-size 64 --save-log logs/sgd_u8_b64
    python main.py  --quantizer qsgd --num-users 8 --batch-size 64 --save-log logs/qsgd_u8_b64
    
