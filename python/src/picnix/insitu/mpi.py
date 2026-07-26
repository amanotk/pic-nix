def allreduce(value, op=None):
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RuntimeError("allreduce requires the picnix mpi extra") from error

    operation = MPI.SUM if op is None else op
    return MPI.COMM_WORLD.allreduce(value, op=operation)
