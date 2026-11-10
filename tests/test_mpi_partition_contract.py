import pytest
from marketintel.mpi_tasks import partitions


def test_partition_every_row_exactly_once():
    for world in (2,3,4,7):
        for n in (0,1,2,7,9,23):
            spans=partitions(n,world)
            all_rows=[j for a,b in spans for j in range(a,b)]
            assert all_rows==list(range(n))
            assert max((b-a for a,b in spans),default=0)-min((b-a for a,b in spans),default=0)<=1


def test_partition_invalid_inputs():
    with pytest.raises(ValueError): partitions(-1,2)
    with pytest.raises(ValueError): partitions(3,0)
