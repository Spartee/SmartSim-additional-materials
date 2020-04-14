#!/bin/env python
import numpy as np
import time

def retrieve_all_ranks(client, key_suffix, nranks, dtype="float64"):
    retrieve_d = {}
    for rank in range(nranks):
        rank_id= f'{rank:06d}'
        key = f'{rank_id}_{key_suffix}'
        retrieve_d[rank_id] = client.get_array_nd_float64(f'{rank_id}_{key_suffix}', wait=True)
    return retrieve_d


def reconstruct_domain(client, timestamp, nranks):
    """Reconstruct the domain of MOM6 by rank"""
    start_iter = time.time()

    meta_rank = retrieve_all_ranks(client, "rank-meta" ,nranks)
    starti_rank  = np.array([int(meta[0]) for meta in meta_rank.values()])
    startj_rank  = np.array([int(meta[2]) for meta in meta_rank.values()])
    glob_starti_rank = starti_rank - min(starti_rank)
    glob_startj_rank = startj_rank - min(startj_rank)

    t1 = time.time()
    h_rank = retrieve_all_ranks(client, f'{timestamp}_T',nranks)

    ni_rank = np.array( [ h.shape[-1] for h in h_rank.values()] )
    nj_rank = np.array( [ h.shape[-2] for h in h_rank.values()] )
    nk_rank = np.array( [ h.shape[-3] for h in h_rank.values()] )

    glob_endi_rank = glob_starti_rank + ni_rank - 1
    glob_endj_rank = glob_startj_rank + nj_rank - 1

    ni_glob = max(glob_endi_rank) + 1 
    nj_glob = max(glob_endj_rank) + 1
    nk_glob = max(nk_rank)

    h_glob = np.zeros([nk_glob,nj_glob,ni_glob])

    for rank in range(nranks):
        si = glob_starti_rank[rank]
        sj = glob_startj_rank[rank]
        ei = glob_endi_rank[rank] + 1
        ej = glob_endj_rank[rank] + 1

        h_glob[:,sj:ej,si:ei] = h_rank[f'{rank:06d}']
    iter_time = time.time() - start_iter
    print(f"Time elapsed in iteration: {iter_time}")
        
    return h_glob
