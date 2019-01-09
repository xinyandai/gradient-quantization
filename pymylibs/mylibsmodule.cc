#include <Python.h>
#include <mpi.h>


struct GlobalState {

  // Whether MPI_Init has been completed
  bool initialization_done;
  // The MPI rank, local rank, and size.
  int rank;
  int size;

  GlobalState(bool done, int r, int s) {
    initialization_done = done;
    rank = r;
    size = s;
  }

};

static GlobalState global(false, 0, 0);


void c_libs_mpi_initialize() {
    if (!global.initialization_done) {
        global.initialization_done = true;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &global.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &global.size);
    }
}

int c_libs_mpi_rank() {
    c_libs_mpi_initialize();
    return global.rank;
}

int c_libs_mpi_size() {
    c_libs_mpi_initialize();
    return global.size;
}

void c_libs_mpi_finalize() {
    if(global.initialization_done) {
        MPI_Finalize();
    }
}


static PyObject* my_libs_mpi_initialize(PyObject* self, PyObject* args)
{
    c_libs_mpi_initialize();
    return Py_None;
}

static PyObject* my_libs_mpi_size(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i", c_libs_mpi_size());
}

static PyObject* my_libs_mpi_rank(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i", c_libs_mpi_rank());
}

static PyObject* my_libs_mpi_finalize(PyObject* self, PyObject* args)
{
    c_libs_mpi_finalize();
    return Py_None;
}


static PyMethodDef methods_list[] = {
    { "mpi_initialize", my_libs_mpi_initialize, METH_NOARGS, "mpi_initialize" },
    { "mpi_finalize", my_libs_mpi_finalize, METH_NOARGS, "mpi_finalize" },
    { "mpi_rank", my_libs_mpi_rank, METH_NOARGS, "mpi_rank" },
    { "mpi_size", my_libs_mpi_size, METH_NOARGS, "mpi_size" },
    { NULL, NULL, 0, NULL }
};


static struct PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "mylibs",
    "MPI Module",
    -1,
    methods_list
};


// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_mylibs(void)
{
    return PyModule_Create(&my_module);
}