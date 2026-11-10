"""Diagnostic: launch using mpiexec -n 4 python scripts/mpi_check.py."""
from mpi4py import MPI
import socket

comm=MPI.COMM_WORLD
ranks=comm.allgather({'rank':comm.rank,'world':comm.size,'host':socket.gethostname()})
if comm.rank==0:
    if comm.size<2 or sorted(x['rank'] for x in ranks)!=list(range(comm.size)):
        raise RuntimeError('Invalid MPI communication or fewer than two ranks')
    print('GENUINE MPI ranks:', ranks)
