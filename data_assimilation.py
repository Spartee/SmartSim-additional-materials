#!/usr/bin/env python
from smartsim import Client
from mom6_da import assimilator
from time import sleep, time
from datetime import datetime
import argparse
import cftime
import xarray as xr
import numpy as np
from pathos.multiprocessing import Pool

class rank_type:

    def __init__(self, id):
        # Set hardcoded values from MOM6
        NIHALO = 4; NJHALO = 4

        self.client = client
        # Set the rank ids
        self.id_int = id
        self.id_str = f'{id:06d}'
        # Set internal and global bounds of arrays on this rank
        self.isd_global, self.jsd_global, self.ks, self.nid, self.njd, self.nk = main_client.get_array_nd_int32(f'{self.id_str}_array-meta')

        # Calculate number of points in the compute domain
        self.nic = self.nid - 2*NIHALO
        self.njc = self.njd - 2*NJHALO
        # Array bounds in the global array
        self.ied_global = self.isd_global + self.nid - 1
        self.jed_global = self.jsd_global + self.njd - 1
        self.isc_global = self.isd_global + NIHALO
        self.jsc_global = self.jsd_global + NJHALO
        self.iec_global = self.isc_global + self.nic - 1
        self.jec_global = self.jsc_global + self.njc - 1
        # Array bounds in the local array
        self.isc_local = NIHALO; self.iec_local = NIHALO + self.nic - 1
        self.jsc_local = NJHALO; self.jec_local = NJHALO + self.njc - 1
        # Allocate the arrays used for the increments
        self.temp_increment= np.zeros((35,self.njd,self.nid))
        self.salt_increment= np.zeros((35,self.njd,self.nid))

    def get_priors(self, da_rank_id):
        self.temp_prior = clients[da_rank_id].get_array_nd_float64(f'{self.id_str}_temp-prior')
        self.salt_prior = clients[da_rank_id].get_array_nd_float64(f'{self.id_str}_salt-prior')

    def send_increments(self, temp_obs, salt_obs, da_rank_id):
        islice_glob = slice(self.isc_global,self.iec_global)
        jslice_glob = slice(self.jsc_global,self.jec_global)
        islice_loc = slice(self.isc_local,self.iec_local)
        jslice_loc = slice(self.jsc_local,self.jec_local)

        self.temp_increment[:,jslice_loc,islice_loc] = (temp_obs[:,jslice_glob,islice_glob] -
                                                        self.temp_prior[:,jslice_loc,islice_loc])
        self.salt_increment[:,jslice_loc,islice_loc] = (salt_obs[:,jslice_glob,islice_glob] -
                                                  self.salt_prior[:,jslice_loc,islice_loc])
        clients[da_rank_id].put_array_nd_float64( f'{self.id_str}_temp-inc', self.temp_increment )
        clients[da_rank_id].put_array_nd_float64( f'{self.id_str}_salt-inc', self.salt_increment )
        clients[da_rank_id].put_scalar_int32( f'{self.id_str}_sent-inc', 1)

    def run_da(self, temp_intp, salt_intp, da_rank):
        da_rank_id = f'{da_rank:06d}'
        print(f"{self.id_str}: Getting prior")
        self.get_priors(da_rank_id)
        print(f"{self.id_str}: Sending increments")
        self.send_increments(temp_intp, salt_intp, da_rank_id)
        clients[da_rank_id].poll_key_and_check_scalar_int32( f'{self.id_str}_sent-prior', 0)
        clients[da_rank_id].put_scalar_int32(f'{self.id_str}_sent-inc',0)
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('temp_da_path', help = 'Path to the temperature file to assimilate')
    parser.add_argument('salt_da_path', help = 'Path the salinity file to assimilate')
    args = parser.parse_args()

    main_client = Client()
    main_client.setup_connections()
    NUM_THREADS = 48
    # Setup SmartSim
    clients = { f'{id:06d}':Client() for id in range(NUM_THREADS) }
    for client in clients:
        clients[client].setup_connections()

    # Initialization of the data asssimilation scripts
    print("Loading observational datasets")
    data = xr.merge([xr.open_dataset(args.temp_da_path).isel({'time':range(12)}).load(),
                     xr.open_dataset(args.salt_da_path).isel({'time':range(12)}).load()])

    # Get simulation metadata
    print("Retrieving simulation metadata")
    simulation_metadata = {}
    main_client.poll_key_and_check_scalar_int32('model-initialized',1)
    simulation_metadata['rank_ids']     = main_client.get_array_nd_int32('rank-ids')
    simulation_metadata['initial_time'] = datetime(*main_client.get_array_nd_int32('initial-time'))
    simulation_metadata['current_time'] = None

    # Reference to 0 time of the 'obs' data
    #shift_time = (data.time.values - data.time.values[0]) + simulation_metadata['initial_time']
    shift_time = (data.time.values - data.time.values[0]) + datetime(2008,1,1)
    print(shift_time, simulation_metadata['initial_time'])
    so_assimilator = assimilator(shift_time,data.so)
    thetao_assimilator = assimilator(shift_time,data.thetao)

    # Setup all the rank classes
    rank_list = [ rank_type(rank_id) for rank_id in simulation_metadata['rank_ids'] ]

    while True:

        print("Beginning DA loop")
        pool = Pool(NUM_THREADS)
        main_client.poll_key_and_check_scalar_int32('sent-time',1)
        current_time = cftime.datetime(*main_client.get_array_nd_int32('simulation-time'))
        print(f"DA Time:{current_time}")
        temp_intp = thetao_assimilator.time_interpolate(current_time)
        salt_intp = so_assimilator.time_interpolate(current_time)

        for rank in rank_list:
            rank.running = False
            main_client.put_scalar_int32(f'{rank.id_str}_sent-inc',0)
        da_todo = [ rank for rank in rank_list if not rank.running ]

        async_list = []
        # Loop over ranks have been processed and add them to the async queue
        da_rank = 0
        while (da_todo):
            for rank in da_todo:
                stime = time()
                if (main_client.poll_key_and_check_scalar_int32(f'{rank.id_str}_sent-prior',1,10,1)):
                    rank.running = True
                    async_list.append( pool.apply_async( rank.run_da, (temp_intp, salt_intp, da_rank) ) )
                    da_rank = (da_rank + 1) % NUM_THREADS
                    # rank.run_da( temp_intp, salt_intp )
                print(time() - stime)
                    

            da_todo = [ rank for rank in rank_list if not rank.running ]
            print(f"Remaining number of ranks: {len(da_todo)}")
        pool.close()
        pool.join()
