#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "mpi.h"
#include <iostream>

using namespace tensorflow;



REGISTER_OP("TfBroadcast")
        .Attr("root: int")
        .Attr("size: int")
        .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Input("buffer: T")
        .Input("pre_node: T")
        .Output("output: T")
        .SetShapeFn(
            [](::tensorflow::shape_inference::InferenceContext* c) {
                c->set_output(0, c->input(0));
                return Status::OK();
            }
        );


REGISTER_OP("TfGather")
        .Attr("root: int")
        .Attr("size: int")
        .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Input("input: T")
        .Input("pre_node: T")
        .Output("output: T")
        .SetShapeFn(
                [](::tensorflow::shape_inference::InferenceContext* c) {
                    int cluster_size;
                    c->GetAttr("size", &cluster_size);
                    ::tensorflow::shape_inference::ShapeHandle output;
                    ::tensorflow::shape_inference::ShapeHandle shape_size = c->Vector(
                        ::tensorflow::shape_inference::DimensionOrConstant(cluster_size));
                    TF_RETURN_IF_ERROR(c->Concatenate(shape_size, c->input(0), &output));
                    c->set_output(0,output);
                    return Status::OK();
                }
        );


MPI_Datatype GetMPIDataType(const Tensor tensor) {
    switch (tensor.dtype()) {
        case DT_UINT8:
            return MPI_UINT8_T;
        case DT_INT8:
            return MPI_INT8_T;
        case DT_UINT16:
            return MPI_UINT16_T;
        case DT_INT16:
            return MPI_INT16_T;
        case DT_INT32:
            return MPI_INT32_T;
        case DT_INT64:
            return MPI_INT64_T;
        case DT_FLOAT:
            return MPI_FLOAT;
        case DT_DOUBLE:
            return MPI_DOUBLE;
        default:
            return 0;
    }
}

class TfBroadcastOp : public OpKernel {
public:
    explicit TfBroadcastOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("root", &root_));
    }
    void Compute(OpKernelContext* context) override {

        // Grab the input tensor

        Tensor input_tensor = context->input(0);
        context->forward_ref_input_to_ref_output(0, 0);

        MPI_Datatype datatype = GetMPIDataType(input_tensor);

        MPI_Bcast(
                (void*)input_tensor.tensor_data().data(),
                (int)input_tensor.NumElements(),
                datatype,
                root_,
                MPI_COMM_WORLD);

        context->set_output(0, input_tensor);
    }

private:
    int root_;
};


class TfGatherOp : public OpKernel {
public:
    explicit TfGatherOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("root", &root_));
        OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
    }
    void Compute(OpKernelContext* context) override {

        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        MPI_Datatype datatype = GetMPIDataType(input_tensor);
        int count = (int)input_tensor.NumElements();

        // Create an output tensor
        Tensor* output_tensor = NULL;

        TensorShape tensor_shape;
        tensor_shape.AddDim(size_);
        tensor_shape.AppendShape(input_tensor.shape());

        OP_REQUIRES_OK(
                context,
                context->allocate_output(0, tensor_shape, &output_tensor)
        );

        MPI_Gather(
                (void*)input_tensor.tensor_data().data(),
                count,
                datatype,
                (void*)output_tensor->tensor_data().data(),
                count,
                datatype,
                root_,
                MPI_COMM_WORLD);
    }

private:
    int root_;
    int size_;
};


REGISTER_KERNEL_BUILDER(Name("TfBroadcast").Device(DEVICE_CPU), TfBroadcastOp);
REGISTER_KERNEL_BUILDER(Name("TfBroadcast").Device(DEVICE_GPU), TfBroadcastOp);

REGISTER_KERNEL_BUILDER(Name("TfGather").Device(DEVICE_CPU), TfGatherOp);
REGISTER_KERNEL_BUILDER(Name("TfGather").Device(DEVICE_GPU), TfGatherOp);
