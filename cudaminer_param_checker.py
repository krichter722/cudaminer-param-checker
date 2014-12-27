#!/usr/bin/python
# -*- coding: utf-8 -*- 

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Dieses Programm ist Freie Software: Sie können es unter den Bedingungen
#    der GNU General Public License, wie von der Free Software Foundation,
#    Version 3 der Lizenz oder (nach Ihrer Wahl) jeder neueren
#    veröffentlichten Version, weiterverbreiten und/oder modifizieren.
#
#    Dieses Programm wird in der Hoffnung, dass es nützlich sein wird, aber
#    OHNE JEDE GEWÄHRLEISTUNG, bereitgestellt; sogar ohne die implizite
#    Gewährleistung der MARKTFÄHIGKEIT oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.
#    Siehe die GNU General Public License für weitere Details.
#
#    Sie sollten eine Kopie der GNU General Public License zusammen mit diesem
#    Programm erhalten haben. Wenn nicht, siehe <http://www.gnu.org/licenses/>.

# see REAMDE.md for an explanation what cuda
# requested https://github.com/cbuchner1/CudaMiner/issues/147 for a parameter 
# which makes `cudaminer --benchmark` produce n hash/s values and exit

import os
import collections
import logging
import subprocess as sp
import time
import itertools
import shutil
import sys
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

cudaminer = "cudaminer"

# default param_dict creation functions
def __launch_config_values__():
    ret_value = []
    itertools_list = itertools.product(
        ["L", "F", ], # <ref>http://www.reddit.com/r/litecoinmining/comments/1t65as/nvidia_kepler_mining_improvements_now_in/</ref>
        [str(i) for i in range(1, 32)], 
        [str(i) for i in range(1, 16)], 
            # L1x17 fails with `cudaminer --benchmark --no-autotune 
            # --hash-parallel 0 --algo scrypt:1 --texture-cache 0 
            # --launch-config L1x17 --single-memory 0
    )
    for itertools_item in itertools_list:
        ret_value.append("%s%sx%s" % (itertools_item[0], itertools_item[1], itertools_item[2]))
    itertools_list = itertools.product(
        ["S", "K", "T", "X"], # <ref>http://www.reddit.com/r/litecoinmining/comments/1t65as/nvidia_kepler_mining_improvements_now_in/</ref>
        [str(i) for i in range(1, 32)], 
        [str(i) for i in range(1, 32)], 
    )
    for itertools_item in itertools_list:
        ret_value.append("%s%sx%s" % (itertools_item[0], itertools_item[1], itertools_item[2]))
    return ret_value
def __scrypt_values__():
    # needs to cover at least scrypt Salsa20/8(1024,1,1), see comment below
    ret_value = ["scrypt:%s" % (str(pow(2, i)),) for i in range(0, 10)]
    return ret_value
def __scrypt_nfactor_values__():
    ret_value = ["scrypt-jane:%s" % (str(pow(2, i)),) for i in range(0, 10)]
    return ret_value
def __scrypt_starttime_values__():
    itertools_list = itertools.product(
        [str(i) for i in range(1, 32)], # StartTime
        [str(i) for i in range(1, 32)], # Nfmin
        [str(i) for i in range(1, 32)], # Nfmax
    )
    ret_value = []
    for itertools_item in itertools_list:
        ret_value.append("scrypt-jane:%s,%s,%s" % (itertools_item[0], itertools_item[1], itertools_item[2]))
    return ret_value

param_dict_default = {
    "--hash-parallel": ["0", "1", "2"], # -H
    "--single-memory": ["0", "1"], # @TODO: check # -m
    "--texture-cache": ["0", "1"], # - " - # -C
    "--launch-config": __launch_config_values__(), # -l
    #"--batchsize": [str(pow(2,i)) for i in range(0,10)], # causes `unrecognized option '--batchsize'`, deactivate temporarily, until clear what `comma separated list of max. scrypt iterations` (from `cudaminer --help`) means # -b
    "--algo": __scrypt_values__()+["scrypt-jane", ]+__scrypt_nfactor_values__()+__scrypt_starttime_values__()+["sha256d", "keccak", "blake", ], 
          # -a
          # `scrypt` can be ommitted because it means `scrypt Salsa20/8(1024,1,1)` which is covered by `scrypt:N` which means `scrypt Salsa 20/8(N,1,1)`
          # unclear what `scrypt-jane:Coin` with `Coin must be one of the supported coins.` (see `cudaminer --help`) means, skipping
}

# invokes `cudaminer` with all combinations of the values in the `param_dict`. 
# The combinations are produced by creating the cartesion product of tuples of 
# each key in `param_dict` and each value in it. `cudaminer` doesn't seem to 
# have an option to specify the number of runs in a benchmark. Therefore the 
# process is simply killed after it printed the first hash/s value to stderr 
# (`cudaminer` prints to stderr for some reason). The check of the output is 
# done every `output_scan_interval` seconds.
# @args param_dict a dictionary in the form of `cudaminer option (long or short)` x 
# `list of option values to test each`. The `=` in the long for of arguments 
# seems to be optional, reported 
# https://github.com/cbuchner1/CudaMiner/issues/148 for clearification and 
# improvement
# @args cudaminer a path to a cudaminer binary
# @args output_scan_interval the time between two checks of the output the 
# program might pause longer if `cudaminer` doesn't return output so fast
# @args output_scan_max_count determines how many times `output_scan_interval` 
# can elapse until the invokation of `cudaminer` is considered as failed and 
# the test run is aborted and skipped
# @args hash_rate_count the number of hash/s rate values which ought to be 
# retrieved from output before the `cudaminer` process is killed
def cudaminer_param_checker(param_dict=param_dict_default, cudaminer=cudaminer, cudaminer_additional_parameters=[], output_scan_interval=1, output_scan_max_count = 120, hash_rate_count=8):
    # don't validate existance and accessibility of cudaminer, see internal 
    # implementation notes below
    if str(type(cudaminer_additional_parameters)) != "<type 'list'>":
        raise ValueError("cudaminer_additional_parameters has to be a list") 
            # ducktyping is nice, but validation is nicer (this is not good 
            # python practice, though)
    
    # create cartesian product in order to acchieve test of all combinations
    itertools_list = []
    for param in param_dict:
        itertools_list_list = []
        param_values = param_dict[param]
        for param_value in param_values:
            itertools_list_list.append((param, param_value))
        itertools_list.append(itertools_list_list)
    param_dict_cartesian = itertools.product(*itertools_list)
    # don't transform itertools result into a collection (e.g. list) because 
    # you loose the iterator abilities
    # count elements in order to notify user about the number (there might be 
    # an itertools function to do this, but not clear yet, rather struggeling 
    # with bad docs, do it yourself:
    param_count = 0
    for param in param_dict:
        param_count += len(param_dict[param])
    logger.info("testing with '%s' parameter combinations" % (str(param_count)))
    result_dict = dict()
    for param_dict_cartesian_item in param_dict_cartesian:
        # breakable nested loops need to be imitated with functions
        def __outer_loop__():
            cmd_tail = []
            for param_tuple in param_dict_cartesian_item:
                cmd_tail.append(param_tuple[0])
                cmd_tail.append(param_tuple[1])
            if len(cudaminer_additional_parameters) > 0:
                logger.debug("invoking cudaminer with requested additional parameters '%s'" % (str(cudaminer_additional_parameters), ))
            cmds = [cudaminer, "--benchmark", "--no-autotune", ]+cudaminer_additional_parameters+cmd_tail
            cmd = str.join(" ", cmds)
            logger.debug("testing '%s'" % (cmd, ))
            cudaminer_process = sp.Popen(cmds, stdout=sp.PIPE, stderr=sp.PIPE) 
                # cudaminer seems to print everything to stderr
            cudaminer_process_output = ""
            output_scan_count = 0
            logger.debug("waiting for cudaminer output containing a hash/s value")
            while cudaminer_process_output.count("hash/s") < hash_rate_count:
                time.sleep(output_scan_interval)
                cudaminer_process_output += str(cudaminer_process.stderr.read(100)) # str conversion necessary in python3
                    # cudaminer produces endless output once it is running and EOF 
                    # when it is terminated (e.g. externally), multiple invokations 
                    # after EOF seem to return ''
                    # @TODO: delete/free preceeding input (we just need to avoid to 
                    # break inside search string `hash/s` (adjust logging message 
                    # below then)            
                output_scan_count += 1
                cudaminer_process_returncode = cudaminer_process.poll()
                if not cudaminer_process_returncode is None:
                    cudaminer_process_output += cudaminer_process.stderr.read() 
                        # read the rest (`read` return immediately when process 
                        # is terminated)
                    logger.warn("cudaminer returned unexpectedly with returncode '%s', consider adjusting param_dict, skipping (output so far has been '%s')" % (str(cudaminer_process_returncode), cudaminer_process_output))
                    return
                if output_scan_count > output_scan_max_count:
                    logger.info("waited longer than '%ss' for cudaminer output, aborting test running and skipping (output so far has been '%s')" % (str(output_scan_count*output_scan_interval), cudaminer_process_output))
                    continue
            cudaminer_process.terminate()
            # retrieve the hash/s value
            cudaminer_process_output_splits = cudaminer_process_output.split("hash/s") # in case the string ends with the split term `''` is added to the split result (which is very smart :))
            hash_rates = []
            for cudaminer_process_output_split in cudaminer_process_output_splits[:-1]: # last item can always be skipped, see above
                cudaminer_process_output_split_split = cudaminer_process_output_split.split(" ")
                hash_rate_suffix = cudaminer_process_output_split_split[-1] # 'k' or 'm' (theoretically others)
                hash_rate = float(cudaminer_process_output_split_split[-2])
                hash_rates.append((hash_rate, hash_rate_suffix))
            logger.debug("results are '%s'" % (str(["%s %s" % (i[0], i[1]) for i in hash_rates])))
            
            # store results for sorted summary (store mean as key for 
            # sorting and keep values in dict value as part of a tuple)
            hash_rate_mean = numpy.mean([i[0] for i in hash_rates])
            result_dict[hash_rate_mean] = (["%s %s" % (i[0], i[1]) for i in hash_rates], cmd)
        __outer_loop__()
        
        # summary
        logger.info("results (ascending):")
        logger.info("hash/s rate mean\thash/s rates\tcommand line")
        for hash_rate in sorted(result_dict):
            logger.info("%s\t%s\t%s" % (str(hash_rate), str(result_dict[hash_rate][0]), str(result_dict[hash_rate][1])))
# internal implementation notes:
# - cross-platform validation of existance and accessibility of the `cudaminer` 
# parameter can only be done with `python3`'s `shutil.which`. Using `python3` 
# cause annoying and unexplainable `/usr/lib/python3.4/subprocess.py:473: 
# ResourceWarning: unclosed file <_io.BufferedReader name=7>  for inst in 
# _active[:]:` for `subprocess.Popen.stderr.read` or `logging.Logger.info`

if __name__ == "__main__":
    cudaminer_param_checker()

