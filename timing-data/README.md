# Timing data description
## Overview
Timings were collected on an XC30 system with an Aries interconnect with a C++/MPI program to put and get arrays to/from a distributed database. Measurements are in seconds calculated based on MPI_walltime calls surrounding each function call. Each column in the spreadsheet aggregates information from 5 trials

## Experiment parameters
The following columns represent the scaling dimensions

- cluster: Number of nodes used for the database cluster
- doubles: How many double precision numbers were sent in a single message
- ppn: Processes per node used for the scaling program
- nodes: Number of nodes to deploy the scaling program
- dpn: Database instances per node

## Timings
The following suffixes represent the reduction operation performed on the rank-by-rank data. Each metric calculated on a trial-by-trial basis and then subsequently averaged over the five trials.

- min: Smallest elapsed time in function
- max: Maximum time elapsed in function
- std: Standard deviation of elapsed time
- mean: Mean elapsed time

A number of functions within the SmartSim stack were timed. Some these are nested inside of others; in the following list functions are listed from the client-facing side down into SmartSim
- put_array<double>: Client-level call to send an array to the database
  - add_array_values: Adds elements from an array to a protobuf array
  - serialize_protobuff: Serialize the array into a character string
  - redis_cluster.set: Higher level routine to manage the putting of an array into the database 
    -- put_to_keydb: Set the serialized string and key in the KeyDB database cluster
- get_array_double<double>: Client-level call to retrieve an array from teh database
  - redis_cluster.get: Retrieve and deserialize array from database

Note: Timings within the paper were reported for the put_array<double> and get_array_double<double> functions 

