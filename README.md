#  quantization for stochastic gradient
## quantization methods
* identical quantization
* scalar quantization
* codebook quantization
## implementation 
Before Run
    `pip install tensorflow`

Run
    `python main.py --num-workers 2 --quantizer identical --two-phases true --network cnn --batch-size 16`
