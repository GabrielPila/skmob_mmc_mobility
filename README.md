# Mobilit Markov Chains with Skmob

This repo is an implementation of Mobility Markov Chains with the Scikit Mobility Library.

It focuses on the use of `Trajectory Data Frames`, a data structure part of skmob, to process geolocated information of an individual in order to generate a Mobility Markov Chain that represents its movement pattern.

# MMC Generation Process

The process is as follows:
1. Loading the gelocated information in a tdf format.

2. Point of Interest Generation (POIs):
  - Noise Filtering: to remove the points in movement
  - Detection of Stops: to identify points where the user spends more than N minutes
  - Compressing: to unify some of the stops and reduce the potential POIs.
  - Clustering: to generate clusters from the stops generated in a DBSCAN-like manner

3. POI assignment, where each point of the original tdf is assigned to a cluster, where possible.

4. Generation of the transition matrix, which shows the transition of the user among different clusters.

5. Generation of the Stationary Vector

# Contributors:
- Gabriel Pila (gabriel.pilah@pucp.edu.pe)
- Anthony Ruiz (ruizc.anthony@gmail.com)
