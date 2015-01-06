`cudaminer_param_checker.py` is a simple attempt to allow users to find out about the parameters for a maximal hash/s rate which can be acchieved with `cudaminer` using `cudaminer`'s benchmark function. The test currently works with a naive brute-force approach which tries the cartesian product of all combinations. A GUI for controlling parameters and visualizing results in a table is provided. Each test takes between 30 and 90 seconds. The script supports saving intermediate results to file and resuming the tests. When you resume a previous run pay attention that the state of the system (I/O and system load) are not too different in order to avoid messing up the results.

## Test parameters
I'm testing the script on a `Lenovo IdeaPad Z-500` with an `NVIDIA GeForce GT740M`. I'll publish my programming result before seriously researching what test parameters acutally mean (there're most certainly several combinations which don't make sense and can be skipped). My experience of up to 800 % of the hash rate I acchieved with specific parameter recommendations speak for themselves. Some value intervals in the test which serve for combinations generation are made up without basic understanding of upper and lower bounds, i.e. some (possibly optimal) value combinations might not be covered by the test.

If you're an expert you might want to use the tool to investigate impact of a parameter or combinations in a certain range.

## Results 
The values in the benchmark are almost never reached during pool mining over `stratum+tcp`. The script results lie at around 140 % of the values experienced during actual pool mining. The results have not yet been evaluated against solo mining. The same device makes 24 Mhash/s with `bitminter` which might indicate that either this script or cudaminer (like I set it up on the test system) or one of its dependencies don't work properly.

## Running
Invoke

    python ./cudaminer_param_checker.py

in the source root. This will run the script with a very large set of 
test parameters which is currently not configurable. Note that depending on your system setup you might need to add 
some commands, e.g. on Ubuntu 14.10 amd64 I need to run 

    env LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH python cudaminer_param_checker.py

after installing the `nvidia` and removing the `nouveau` driver and installing the NVIDIA CUDA toolkit according to http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html (note the hints in the instructions regarding i386 systems). The invokation of extra programs, like `optirun`, can be configured as prepended command within the GUI.

Some GUI elements simply don't have any effect at all and are ignored. The console GUI is currently unmaintained and therefore might not work.

## Prerequisites
Install the `wx` python bindings, e.g. on Ubuntu with `sudo apt-get install python-wxgtk3.0`. `python setup.py build && sudo python setup.py install` will do the rest.

## Internals
The script uses the python `itertools` and `shelve` package which allows quite efficient memory usage, therefore an almost arbitrary large set of parameters won't cause a crash of the application or worse your system due to missing memory - but the time it takes to run all the tests can easily go up to weeks and months.

The script does nothing but parsing the output of `cudaminer --benchmark --no-autotune` with the test arguments which leaves a lot of room for improvement, especially regarding integration with the `cudaminer` source directly.

