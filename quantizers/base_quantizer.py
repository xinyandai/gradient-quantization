from .ps_quantizer import PSQuantizer
from .ring_quantizer import RingQuantizer


def Quantizer(Compressor, parameters, args):
    if args.mode == 'ps':
        return PSQuantizer(Compressor, parameters, args)
    elif args.mode == 'ring':
        return RingQuantizer(Compressor, parameters, args)
    assert False, "mode {} not recognized".format(args.mode)
