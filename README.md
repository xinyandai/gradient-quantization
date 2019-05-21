#  quantization for stochastic gradient

Run HSQ(d=16, m=2^8=256, 6bit for norm quantization)

    python main.py  --quantizer hsq  --network resnet50 --dataset cifar10 \
    --c-dim 16  --k-bit 8 --n-bit 6 --num-users 8 --batch-size 32 --logdir logs_hsq  --log-epoch 10
    python main.py  --quantizer hsq  --network fcn --dataset mnist \
    --c-dim 16  --k-bit 8 --n-bit 6 --num-users 8 --batch-size 32 --logdir logs_hsq  --log-epoch 10
Run SGD
 
    python main.py  --quantizer sgd  --network resnet50 --dataset cifar10 \
    --num-users 8 --batch-size 32 --logdir logs_sgd  --log-epoch 10
    python main.py  --quantizer sgd  --network fcn --dataset mnist \
    --num-users 8 --batch-size 32 --logdir logs_sgd  --log-epoch 10
Run QSGD(2bit, d=128)

    python main.py  --quantizer qsgd --network resnet50 --dataset cifar10 \
    --c-dim 128 --n-bit 2 --num-users 8 --batch-size 32 --logdir logs_qsgd  --log-epoch 10
    python main.py  --quantizer qsgd --network fcn --dataset mnist \
    --c-dim 128 --n-bit 2 --num-users 8 --batch-size 32 --logdir logs_qsgd --log-epoch 10
Run TernGrad(1bit-QSGD, d=#parameters in each layer )

    python main.py  --quantizer qsgd --network resnet50 --dataset cifar10 \
    --c-dim 0 --n-bit 1 --num-users 8 --batch-size 32 --logdir logs_terngrad  --log-epoch 10
    python main.py  --quantizer qsgd --network fcn --dataset mnist \
    --c-dim 0 --n-bit 1 --num-users 8 --batch-size 32 --logdir logs_terngrad --log-epoch 10
Run SignSGD

    python main.py  --quantizer sign --network resnet50 --dataset cifar10 \
    --num-users 8 --batch-size 32 --logdir logs_signsgd  --log-epoch 10
    python main.py  --quantizer sign --network fcn --dataset mnist \
    --num-users 8 --batch-size 32 --logdir logs_signsgd --log-epoch 10
