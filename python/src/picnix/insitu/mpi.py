def allreduce(value, comm=None, op=None):
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RuntimeError("allreduce requires the picnix mpi extra") from error

    communicator = MPI.COMM_WORLD if comm is None else comm
    operation = MPI.SUM if op is None else op
    return communicator.allreduce(value, op=operation)
