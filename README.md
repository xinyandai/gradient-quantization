#  quantization for stochastic gradient
## quantization methods
* sgd
* qsgd
* hsq
* nnq
## implementation 
Run

    python main.py  --quantizer sgd  --network resnet18 --dataset cifar10 --num-users 8 --batch-size 64 
    python main.py  --quantizer qsgd --network resnet18 --dataset cifar10 --c-dim 512 --n-bit 8 --num-users 8 --batch-size 64 
    python main.py  --quantizer hsq  --network resnet18 --dataset cifar10 --c-dim 32  --k-bit 8 --n-bit 8 --num-users 8 --batch-size 64 
    python main.py  --quantizer nnq  --network resnet18 --dataset cifar10 --c-dim 32  --k-bit 8 --n-bit 8 --num-users 8 --batch-size 64 
    
