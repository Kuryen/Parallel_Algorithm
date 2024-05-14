import time
import networkx as nx
import gzip
from mpi4py import MPI


def SCC(iGraph):
    CCTable = {}
    n = len(iGraph) - 1
    for node in iGraph.nodes:
        DTable = nx.shortest_path_length(iGraph, source=node)
        CCTable[node] = n / sum(DTable.values())
    return CCTable


def PCC(iGraph, comm, outfile):
    rank = comm.Get_rank()
    size = comm.Get_size()
    WList = list(iGraph.nodes)[rank::size]
    PCCTable = {}
    n = len(iGraph) - 1
    for node in WList:
        DTable = nx.shortest_path_length(iGraph, source=node)
        PCCTable[node] = n / sum(DTable.values())
    CCTable = comm.gather(PCCTable, root=0)

    if rank == 0:
        combined_centrality = {}
        for cent_dict in CCTable:
            combined_centrality.update(cent_dict)

        outfile.write(f"Number of processes: {size}\n")
        outfile.write("Closeness Centrality Table:\n")
        for node, centrality in combined_centrality.items():
            outfile.write(f"{node}: {centrality}\n")

        outfile.write("\n")


if __name__ == '__main__':
    GraphOne = nx.Graph()
    GraphTwo = nx.Graph()

    with gzip.open('twitter_combined.txt.gz', 'rt') as f:
        for line in f:
            edge = line.strip().split()
            GraphOne.add_edge(edge[0], edge[1])

    with gzip.open('facebook_combined.txt.gz', 'rt') as f:
        for line in f:
            edge = line.strip().split()
            GraphTwo.add_edge(edge[0], edge[1])

    GList = {'Twitter Dataset': GraphOne, 'Facebook Dataset': GraphTwo}

    num_processes = [2, 4, 8, 16, 32, 64]

    for Key in GList.keys():

        with open('output.txt', 'w') as outfile:
            outfile.write(f"{Key}\n")

            ts = time.time()
            CCTableOne = SCC(GList[Key])
            tf = time.time()
            outfile.write("Serial Algorithm\n")
            outfile.write("Closeness Centrality Table:\n")
            for node, centrality in CCTableOne.items():
                outfile.write(f"{node}: {centrality}\n")
            ST = tf - ts
            outfile.write(f"Time taken for Serial Algorithm: {ST} seconds\n\n")

            for np in num_processes:
                ts = time.time()
                comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() // np)
                PCC(GList[Key], comm, outfile)
                tf = time.time()
                PT = tf - ts
                SU = (ST - PT) / ST
                outfile.write(f"Time taken for closeness centrality with {np} processes: {PT} seconds\n Speed Up: {SU} \n\n")


