from quantizers import IdenticalQuantizer
import numpy as np

def test_identical():
    placeholders = [0 for _ in range(3)]
    quantizer = IdenticalQuantizer(placeholders)
    gradient = [1., 2., 3., ]
    compressed = quantizer.encode(gradient)
    compressed_gradients = [compressed, compressed]
    decompressed_gradients = quantizer.decode(compressed_gradients) 
    print(decompressed_gradients)
    print(decompressed_gradients == gradient)
    assert compressed == gradient
    assert list(decompressed_gradients) == gradient
