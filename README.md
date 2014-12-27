`cudaminer_param_checker.py` is a simple attempt to allow users to find out about the parameters for a maximal hash/s rate which can be acchieved with `cudaminer` using `cudaminer`'s benchmark function. The test currently works with a naive brute-force approach which tries the cartesian product of all combinations. Currently there's a test case which passed parameters which cause > 32000 combinations to be tested. Each test takes between 30 and 90 seconds. After all tests ran a sorted list of the hash rates and corresponding `cudaminer` parameters is printed to console.

## Test parameters
I'm testing the script on a `Lenovo IdeaPad Z-500` with an `NVIDIA GeForce GT740M`. I'll publish my programming result before seriously researching what test parameters acutally mean (there're most certainly several combinations which don't make sense and can be skipped). My experience of up to 800 % of the hash rate I acchieved with specific parameter recommendations speak for themselves. Some value intervals in the test which serve for combinations generation are made up without basic understanding of upper and lower bounds, i.e. some (possibly optimal) value combinations might not be covered by the test.

If you're an expert you might want to use the tool to "visualize" - the visualization is currently so poor that it can be considered inexistent; I phantasized about tables and graphs, we'll see what that brings - impact of a parameter or a combinations in a certain range

## Internals
The script uses the python `itertools` package which allows quite efficient memory usage, therefore an almost arbitrary large set of parameters won't cause a crash of the application or worse your system due to missing memory - but the time it takes to run all the tests can easily go up to weeks and months.

The script does nothing but parsing the output of `cudaminer --benchmark --no-autotune` which leaves a lot of room for improvement, especially regarding integration with the `cudaminer` source directly.

## Results 
The values in the benchmark are almost never reached during pool mining over `stratum+tcp`. The script results lie at around 140 % of the values experienced during actual pool mining. The results have not yet been evaluated against solo mining.

## Running
Invoke

    python ./tests/cudaminer_param_checker_test.py

in the source root. This will run the script with a very large set of 
test parameters before or after some uninteresting tests. Note that depending on your system setup you might need to add 
some commands, e.g. on Ubuntu 14.10 amd64 I need to run 

    optirun env LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH python tests/cudaminer_param_checker_test.py

after installing the `nvidia` and removing the `nouveau` driver and installing the NVIDIA CUDA toolkit according to http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html (not the hints in the instructions regarding i386 systems).

