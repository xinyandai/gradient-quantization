mypython="/home/xinyan/anaconda3/bin/python"
TF_CFLAGS=( $($mypython -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($mypython -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ \
-std=c++11 \
-I/usr/local/include \
-shared mpi_ops.cc \
-o mpi_ops.so \
-pthread \
-Wl,-rpath \
-Wl,/usr/local/lib \
-Wl,--enable-new-dtags \
-lmpi \
-L/usr/local/lib \
-fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
-O2
