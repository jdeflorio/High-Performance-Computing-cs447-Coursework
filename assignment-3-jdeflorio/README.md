# assignment-3-jdeflorio

This assignment was to show the fact that in high-dimensional space most of the volume of a sphere lies near the surface of the sphere. 

To do this a random sample of uniformly distributed points within the volume of a unit sphere was created, and then its distance from the center was computed.

The [rejection method](https://en.wikipedia.org/wiki/Rejection_sampling) was used to generate the points within the hypersphere. 

For the parallelization, the OpenMP interface was used for the loop-level threading to generating the points.
