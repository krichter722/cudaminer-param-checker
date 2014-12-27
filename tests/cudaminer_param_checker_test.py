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

# parameter bounds are currently unclear, asked 
# http://bitcoin.stackexchange.com/questions/34193/which-are-value-bounds-of-
# cudaminers-launch-config-batchsize-texture-cache-s for inputs

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

