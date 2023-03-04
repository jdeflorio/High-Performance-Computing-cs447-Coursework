# assignment-2-jdeflorio

This assignment was to parallelize the [k-NN problem](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) using threading.

Training and Query points were given, as well as scripts to generate/dump out their contents.

To solve this problem a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) was created and then queried using parallelization.

The main take away from this assignment was working with job queues/worker threads and correctly locking/unlocking when needed.
