#  quantization for stochastic gradient
## quantization methods
* identical quantization
* scalar quantization
* codebook quantization
## implementation 
Before Run

    pip install ray
    pip install tensorflow
Run Ray

    python ray_syn.py --quantizer codebook --num-workers 4  --two-phases True
