#!/usr/bin/python

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

import unittest
import itertools
import logging
import sys
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.path.append(os.path.realpath(os.path.join(__file__, "..", "..")))
import cudaminer_param_checker

class CudaminerParamCheckerTest(unittest.TestCase):

    def test_cudaminer_param_checker_full_run(self):
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
        
        param_dict = {
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
        logger.info("testing with param_dict '%s'" % (str(param_dict),))
        cudaminer_param_checker.cudaminer_param_checker(param_dict)
    
    # test program output, failing invokations of `cudaminer` and result output
    def test_cudaminer_param_checker_flow(self):
        # test sorting with 4 results
        param_dict = {"--hash-parallel": ["0", "1"], 
            "--algo": ["scrypt:1"], 
            "--texture-cache": ["0", "1"], 
            "--launch-config": ["L9x8"], 
            "--single-memory": ["0"], 
        }
        cudaminer_param_checker.cudaminer_param_checker(param_dict)
        # test skipping of failing `cudaminer` process
        param_dict = {"--hash-parallel": ["0"], 
            "--algo": ["scrypt:1"], 
            "--texture-cache": ["0"], 
            "--launch-config": ["L1x17"], 
            "--single-memory": ["0"], 
        }
        cudaminer_param_checker.cudaminer_param_checker(param_dict) 

if __name__ == '__main__':
    unittest.main()

