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

# see REAMDE.md for an explanation what cudaminer-param-checker does

# internal implementation notes:
# - requested https://github.com/cbuchner1/CudaMiner/issues/147 for a parameter 
# which makes `cudaminer --benchmark` produce n hash/s values and exit
# - the idea to add a value after generation has been canceled makes the GUI 
# appear unresponsive, just skip it
# - in order to share the information whether the value generation is running 
# or not in a multithreaded enviroment, the corresponding flag has to be 
# handled in a class and therefore all code has to be object oriented 
# (a global flag declared at the top level of the script isn't writable from 
# within and outside of the `wx.Frame` subclass) -> a GUI has to be object 
# oriented and if one part of the script is, it's much easier to make 
# everything object oriented
# - in order to deal with memory consuming resume-capable storage of results, 
# use the `shelve` module (file-backed dictionaries). The fact that this might 
# require available disk space when storing into a temporary file until 
# explicit saving of results is requested doesn't matter (due to the fact that all information needs to be preserved in order restore the summary grid when generation is resumed, it is not sufficient to limit the number of values displayed in the grid; although saving items in the form of tuples of a dict saves up to 7/8 of the memory needs, it is still poor and switching to shelve is a rather small step with great advantages)
# - due to the fact that `shelve` doesn't support arbitrary keys, the `dict` keys are serialized using `marshal.dumps` (consider using `lru_cache` function annotation and therefore switching to python3 (see https://docs.python.org/3.4/library/functools.html#functools.lru_cache for details))

import os
import collections
import logging
import subprocess as sp
import time
import itertools
import shutil
import sys
import numpy
import texttable
import progressbar
import wx
import wx.grid
import wx.lib
import wx.lib.scrolledpanel
import python_essentials
import python_essentials.lib
import python_essentials.lib.os_utils as os_utils
import cudaminer_param_checker_globals
import threading
import plac
import psutil
import signal
import shelve
import marshal
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

cudaminer_default = os_utils.which("cudaminer")

FRONTEND_CONSOLE_GUI = "text-gui"
FRONTEND_WXPYTHON_GUI = "wxpython-gui"
frontends = [FRONTEND_CONSOLE_GUI, FRONTEND_WXPYTHON_GUI]
frontend_default = FRONTEND_WXPYTHON_GUI

hash_rate_count_default = 8
output_scan_max_count_default = 1200
output_scan_interval_default = 0.1
debug_default = False

__cudaminer_docstring__ = "a path to a cudaminer binary"
__output_scan_interval_docstring__ = "the time between two checks of the output the program might pause longer if `cudaminer` doesn't return output so fast"
__output_scan_max_count_docstring__ = "determines how many times `output_scan_interval` can elapse until the invokation of `cudaminer` is considered as failed and the test run is aborted and skipped"
__hash_rate_count_docstring__ = "the number of hash/s rate values which ought to be retrieved from output before the `cudaminer` process is killed"

# default param_dict creation functions
def __launch_config_values__():
    ret_value = []
    itertools_list = itertools.product(
        ["L", "F", ], # <ref>http://www.reddit.com/r/litecoinmining/comments/1t65as/nvidia_kepler_mining_improvements_now_in/</ref>
        [i for i in range(1, 32)], 
        [i for i in range(1, 16)], 
            # L1x17 fails with `cudaminer --benchmark --no-autotune 
            # --hash-parallel 0 --algo scrypt:1 --texture-cache 0 
            # --launch-config L1x17 --single-memory 0
    )
    for itertools_item in itertools_list:
        ret_value.append("%s%dx%d" % (itertools_item[0], itertools_item[1], itertools_item[2]))
    itertools_list = itertools.product(
        ["S", "K", "T", "X"], # <ref>http://www.reddit.com/r/litecoinmining/comments/1t65as/nvidia_kepler_mining_improvements_now_in/</ref>
        [i for i in range(1, 32)], 
        [i for i in range(1, 32)], 
    )
    for itertools_item in itertools_list:
        ret_value.append("%s%dx%d" % (itertools_item[0], itertools_item[1], itertools_item[2]))
    return ret_value
def __scrypt_values__():
    # needs to cover at least scrypt Salsa20/8(1024,1,1), see comment below
    ret_value = ["scrypt:%d" % (pow(2, i),) for i in range(0, 10)]
    return ret_value
def __scrypt_nfactor_values__():
    ret_value = ["scrypt-jane:%d" % (pow(2, i),) for i in range(0, 10)]
    return ret_value
def __scrypt_starttime_values__():
    itertools_list = itertools.product(
        [i for i in range(1, 32)], # StartTime
        [i for i in range(1, 32)], # Nfmin
        [i for i in range(1, 32)], # Nfmax
    )
    ret_value = []
    for itertools_item in itertools_list:
        ret_value.append("scrypt-jane:%d,%d,%d" % (itertools_item[0], itertools_item[1], itertools_item[2]))
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

def __storage_file_path_default__():
    ret_value = os.path.join(tempfile.mkdtemp(), "cudaminer-param-checker.tmp")
    return ret_value

class CudaminerParamCheckerGenerator():
    # internal implementation notes:
    # - the cudaminer process is not exposed because it's more elegant
    # - in order to be able to skip all already generated values when loading intermediate results from file the intermediate values need to be loaded completely
    
    def __init__(self, param_dict=param_dict_default, storage_file_path=__storage_file_path_default__()):
        """@args storage_file_path a file path for the `shelve` data to be stored. If it points to an existing `shelve` file, the generation for stored keys will be skipped (including invokation of `update_callback` in the `generate_cudaminer_param_checker_values` function; all other setup needs to be done outside of the `CudaminerParamCheckerGenerator`. Passing `tempfile.mkstemp()[1]` causes `error: db type could not be determined` which nonsense, reported as http://bugs.python.org/issue23174"""
        self.generationRunning = False
        self.cudaminerProcess = None
        self.result_dict= shelve.open(storage_file_path) 
        self.storage_file_path = storage_file_path
        
        # create cartesian product in order to acchieve test of all combinations
        itertools_list = []
        for param in param_dict:
            itertools_list_list = []
            param_values = param_dict[param]
            for param_value in param_values:
                itertools_list_list.append((param, param_value))
            itertools_list.append(itertools_list_list)
        self.param_dict_cartesian = itertools.product(*itertools_list)
        # don't transform itertools result into a collection (e.g. list) because 
        # you loose the iterator abilities
        # count elements in order to notify user about the number (there might be 
        # an itertools function to do this, but not clear yet, rather struggeling 
        # with bad docs, do it yourself:
        self.param_count_current = len(self.result_dict)
        self.param_count_max = 0
        for param in param_dict:
            self.param_count_max += len(param_dict[param])
    
    def getResultDict(self):
        return self.result_dict
    
    def getStorageFilePath(self):
        return self.storage_file_path
    
    def clear(self):
        self.result_dict.clear()
    
    def getProgressMax(self):
        """The number of values to be tested during the complete run"""
        return self.param_count_max
    
    def getProgressCurrent(self):
        """The number of values which have already been tested"""
        return self.param_count_current
    
    def start(self):
        self.generationRunning = True
    
    def stop(self):
        self.generationRunning = False
        if not self.cudaminerProcess is None:
            self.cudaminerProcess.terminate()
        self.result_dict.close()
    
    def isRunning(self):
        return self.generationRunning
    
    def generate_cudaminer_param_checker_values(self, cudaminer=cudaminer_default, cudaminer_parameters_prepend=[], cudaminer_additional_parameters=[], output_scan_interval=output_scan_interval_default, output_scan_max_count=output_scan_max_count_default, hash_rate_count=hash_rate_count_default, update_callback=None, check_running_callback=None, ):
        """invokes `cudaminer` with all combinations of the values in the `param_dict`. The combinations are produced by creating the cartesion product of tuples of each key in `param_dict` and each value in it. `cudaminer` doesn't seem to have an option to specify the number of runs in a benchmark. Therefore the process is simply killed after it printed the first hash/s value to stderr (`cudaminer` prints to stderr for some reason). The check of the output is done every `output_scan_interval` seconds.
        @args cudaminer %(__cudaminer_docstring__)s
        @args output_scan_interval %(__output_scan_interval_docstring__)s
        @args output_scan_max_count %(__output_scan_max_count_docstring__)s
        @args hash_rate_count %(__hash_rate_count_docstring__)s
        @args param_dict a dictionary in the form of `cudaminer option (long or short)` x `list of option values to test each`. The `=` in the long for of arguments seems to be optional, reported as https://github.com/cbuchner1/CudaMiner/issues/148 for clearification and improvement
        @args cudaminer_parameters_prepend a list of strings which represents a command or a list of commands which is prepended to the `cudaminer` command (e.g. `["optirun"]` in case you're using the `bumblebee` program). The resulting process tree is killed recursively by sending `SIGTERM`. Make sure that the command accepts it and exits after receiving it, otherwise you risk that the invokation with timeout systematically (see documentation of `output_scan_max_count` for details) and not produce any results.
        @args cudaminer_additional_parameters a list of strings representing `cudaminer` arguments, see `cudaminer --help` for available arguments
        @args update_callback a callable which called with a tuple consisting of a float ranging between `0` and `1` representing the progress of the generation and a tuple representing the newly added tuple to the return value. In case the `cudaminer` process fails the invokation of `update_callback` is skipped
        @args check_running_callback a callable which is invoked regularily and should return `True` if the generation can continue to run and should return `False` if not. If `check_running_callback` is `None` the check doesn't occur""" %  {"__cudaminer_docstring__": __cudaminer_docstring__, "__output_scan_interval_docstring__": __output_scan_interval_docstring__, "__output_scan_interval_docstring__": __output_scan_interval_docstring__, "__output_scan_max_count_docstring__": __output_scan_max_count_docstring__, "__hash_rate_count_docstring__": __hash_rate_count_docstring__}
        # internal implementation notes:
        # - `result_dict_skip` needs to contain the measured values in order to be able 
        # to complete the table when the generation is resumed
        if os_utils.which(cudaminer) is None:
            raise ValueError("cudaminer binary '%s' doesn't exist or isn't accessible, aborting" % (cudaminer, ))
        if str(type(cudaminer_parameters_prepend)) != "<type 'list'>":
            raise ValueError("cudaminer_parameters_prepend has to be a list") 
                # ducktyping is nice, but validation is nicer (this is not good 
                # python practice, though)
        if str(type(cudaminer_additional_parameters)) != "<type 'list'>":
            raise ValueError("cudaminer_additional_parameters has to be a list")
        if output_scan_interval > 0.5:
            logger.warn("an output_scan_interval above 0.5 (specified '%d') is strongly discouraged because it makes the application non-responsive for at least(!) that amount of time" % (output_scan_interval, ))
        
        logger.info("testing with '%d' parameter combinations, reusing results of '%d' combinations from resumption" % (self.param_count_max-len(self.result_dict), len(self.result_dict), ))
        # breakable nested loops need to be imitated with functions
        # @return `True` when `result_dict` has been updated, `False` otherwise
        def __outer_loop__(param_dict_cartesian_item):
            result_dict_key = __generate_result_dict_key__(param_dict_cartesian_item)
            result_dict_key_shelve = __marshal_shelve_key__(result_dict_key)
            if result_dict_key_shelve in self.result_dict:
                return False
        
            cmd_tail = []
            for param_tuple in param_dict_cartesian_item:
                cmd_tail.append(param_tuple[0])
                cmd_tail.append(param_tuple[1])
            if len(cudaminer_parameters_prepend) > 0:
                logger.debug("invoking cudaminer with requested prepended commands '%s'" % (str(cudaminer_parameters_prepend), ))
            if len(cudaminer_additional_parameters) > 0:
                logger.debug("invoking cudaminer with requested additional parameters '%s'" % (str(cudaminer_additional_parameters), ))
            cmds = cudaminer_parameters_prepend+[cudaminer, "--benchmark", "--no-autotune", ]+cudaminer_additional_parameters+cmd_tail
            cmd = str.join(" ", cmds)
            logger.debug("testing '%s'" % (cmd, ))
            self.cudaminerProcess = sp.Popen(cmds, stdout=sp.PIPE, stderr=sp.PIPE)
                # cudaminer seems to print everything to stderr
                # due to the fact that preceeding commands don't necessarily 
                # forward `SIGTERM`, e.g. `optirun` holds back `SIGTERM`, kill 
                # the process tree recursively when it is terminated (see below)
            cudaminer_process_output = ""
            output_scan_count = 0
            logger.debug("waiting for cudaminer output containing a hash/s value")
            while cudaminer_process_output.count("hash/s") < hash_rate_count:
                time.sleep(output_scan_interval)
                cudaminer_process_output += str(self.cudaminerProcess.stderr.read(100)) # str conversion necessary in python3
                    # cudaminer produces endless output once it is running and EOF 
                    # when it is terminated (e.g. externally), multiple invokations 
                    # after EOF seem to return ''
                    # @TODO: delete/free preceeding input (we just need to avoid to 
                    # break inside search string `hash/s` (adjust logging message 
                    # below then)            
                output_scan_count += 1
                cudaminer_process_returncode = self.cudaminerProcess.poll()
                if not self.generationRunning:
                    # hard termination requested
                    kill_process_recursively(self.cudaminerProcess.pid)
                    return False # return codes for this function need to be 
                        # introduced if more complex code is inserted after the 
                        # call below
                if not cudaminer_process_returncode is None:
                    cudaminer_process_output += self.cudaminerProcess.stderr.read() 
                        # read the rest (`read` return immediately when process 
                        # is terminated)
                    logger.warn("cudaminer returned unexpectedly with returncode '%s', consider adjusting param_dict, skipping (output so far has been '%s')" % (str(cudaminer_process_returncode), cudaminer_process_output))
                    return False
                if output_scan_count > output_scan_max_count:
                    logger.info("waited longer than '%d' seconds for cudaminer output, aborting test running and skipping (output so far has been '%s')" % ((output_scan_count*output_scan_interval), cudaminer_process_output))
                    continue
            kill_process_recursively(self.cudaminerProcess.pid)
            # retrieve the hash/s value
            cudaminer_process_output_splits = cudaminer_process_output.split("hash/s") # in case the string ends with the split term `''` is added to the split result (which is very smart :))
            hash_rates_list = []
            for cudaminer_process_output_split in cudaminer_process_output_splits[:-1]: # last item can always be skipped, see above
                cudaminer_process_output_split_split = cudaminer_process_output_split.split(" ")
                hash_rate_suffix = cudaminer_process_output_split_split[-1] # 'k' or 'm' (theoretically others)
                hash_rate = float(cudaminer_process_output_split_split[-2])
                hash_rates_list.append((hash_rate, hash_rate_suffix))
            logger.debug("results are '%s'" % (str(["%s %s" % (i[0], i[1]) for i in hash_rates_list])))
            
            # store results for sorted summary (store mean as key for 
            # sorting and keep values in dict value as part of a tuple)
            hash_rate_mean = numpy.mean([i[0] for i in hash_rates_list])
            self.result_dict[result_dict_key_shelve] = (hash_rate_mean, tuple(["%s %s" % (i[0], i[1]) for i in hash_rates_list])) # storing in tuple makes in possible to store in dict (list are not hashable)
            return True
        for param_dict_cartesian_item in self.param_dict_cartesian:
            if not check_running_callback is None and not check_running_callback():
                logger.info("check_running_callback triggered interruption of generation process, returning intermediate result(s)")
                break
            result_dict_updated = __outer_loop__(param_dict_cartesian_item)
            if result_dict_updated:
                self.param_count_current += 1
                update_callback(self.param_count_current, self.param_count_max)

def kill_process_recursively(parent_pid, sig=signal.SIGTERM, include_parent=True):
    """Kills all child processes of the process with pid `parent_pid` and the process with pid `parent_pid` if `include_parent` is `True`.
    @args include_parent kill process with pid `parent_pid` as well
    @raise psutil.error.NoSuchProcess in `parent_pid` doesn't refer to an existing process"""
    p = psutil.Process(parent_pid)
    child_pids = p.get_children(recursive=True)
    for child_pid in child_pids:
        os.kill(child_pid.pid, sig)
    if include_parent:
        os.kill(parent_pid, sig)

def __generate_result_dict_key__(param_dict_cartesian_item):
    ret_value = dict()
    for param_tuple in param_dict_cartesian_item:
        ret_value[param_tuple[0]] = param_tuple[1]
    return ret_value

def __marshal_shelve_key__(key):
    ret_value = marshal.dumps(key)
    return ret_value

def __unmarshal_shelve_key__(key):
    ret_value = marshal.loads(key)
    return ret_value

def visualize_cudaminer_param_checker_results_console_gui(cudaminer=cudaminer_default, output_scan_interval=output_scan_interval_default, output_scan_max_count=output_scan_max_count_default, hash_rate_count=hash_rate_count_default):
    """visualizes results of a run of cudaminer in a table on the console
    @args result_dict as returned by `generate_cudaminer_param_checker_values`"""
    generator = CudaminerParamCheckerGenerator()
    pbar = progressbar.ProgressBar(maxval=1000).start() # setting maxval to 1000 makes it possible to scale the progress to .1 percent
    def __update_callback__(value):
        pbar.update(pbar_progress*1000)
    generator.start()
    result_dict = generator.generate_cudaminer_param_checker_values(cudaminer=cudaminer, output_scan_interval=output_scan_interval, output_scan_max_count=output_scan_max_count, hash_rate_count=hash_rate_count, update_callback=__update_callback__)
    pbar.finish()
    logger.info("results (ascending):")
    summary_table = texttable.Texttable()
    summary_table.set_cols_align(["l", "l", "l"])
    summary_table.set_cols_valign(["t", "t", "t"])
    summary_table.add_rows([["hash/s rate mean", "hash/s rates", "command line"]], header=True)
    for hash_rate in sorted(result_dict):
        summary_table.add_row([hash_rate, str(result_dict[hash_rate][0]), str(result_dict[hash_rate][1])])
    logger.info(summary_table.draw())

def visualize_cudaminer_param_checker_results_wxpython_gui():
    """there's no need to accept command line arguments (it's possible, but the usecase where initial values for GUI components are specified on command line is very improbable, therefore don't cover it)"""
    app = wx.App()
    frame = CudaminerParamChecker(None, )
    frame.Show()
    app.MainLoop()

class CudaminerParamChecker(wx.Frame):
    """A GUI which allows a resumable generation of hash/s rates of cudaminer. The generation is done with instances of `CudaminerParamCheckerGenerator` which store values in a `shelve` dictionary. This allows easy resumption of the generation."""    
    
    def __init__(self, parent):
        # First, call the base class' __init__ method to create the frame
        wx.Frame.__init__(self, parent, title="%s (%s)" % (cudaminer_param_checker_globals.app_name, cudaminer_param_checker_globals.app_version_string, ), pos=(100, 100), size=(500, 600))
        # frame event binding
        self.Bind(wx.EVT_CLOSE, self.onCloseWindow)
        frameSizer = wx.BoxSizer(wx.VERTICAL) # seems to be necessary for the `CollapsiblePane`s to work properly, see documentation enhancement request at http://www.wxpython.org/Phoenix/docs/html/CollapsiblePane.html as well
        self.generator = CudaminerParamCheckerGenerator()
        self.cudaminer = cudaminer_default
        self.hashRateCount = hash_rate_count_default
        self.numberOfCandidates = 10 # 0 means unlimited
        # main panel setup
        self.panel = wx.Panel(self, -1)
        panelBoxWidget= wx.BoxSizer(wx.VERTICAL)
        # cudaminer controls
        cudaminerBinaryLabelWidget = wx.StaticText(self.panel, label="Path to cudaminer binary:")
        cudaminerBinaryFileChooserWidget = wx.FilePickerCtrl(self.panel, path=self.cudaminer)
        cudaminerBinaryBoxWidget = wx.BoxSizer(wx.HORIZONTAL)
        cudaminerBinaryBoxWidget.Add(cudaminerBinaryLabelWidget, 0)
        cudaminerBinaryBoxWidget.Add(cudaminerBinaryFileChooserWidget, 1, wx.EXPAND)
        panelBoxWidget.Add(cudaminerBinaryBoxWidget, flag=wx.EXPAND|wx.LEFT|wx.TOP|wx.BOTTOM|wx.RIGHT, border=15)
        cudaminerParametersCollapsiblePaneWidget = wx.CollapsiblePane(self.panel, label="Additional cudaminer parameters:")
        cudaminerParametersBoxWidget = wx.GridBagSizer(hgap=5, vgap=5)
        cudaminerParametersPrependLabel = wx.StaticText(cudaminerParametersCollapsiblePaneWidget.GetPane(), label="Prepend additional commands to cudaminer command:")
        self.cudaminerParametersPrependTextCtrl = wx.TextCtrl(cudaminerParametersCollapsiblePaneWidget.GetPane(), value="optirun")
        cudaminerParametersLabelWidget = wx.StaticText(cudaminerParametersCollapsiblePaneWidget.GetPane(), label="Append additional cudaminer parameters:")
        self.cudaminerParametersTextCtrl = wx.TextCtrl(cudaminerParametersCollapsiblePaneWidget.GetPane(), value="")
        self.cudaminerParametersPrependTextCtrl.Bind(wx.EVT_TEXT, self.onCudaminerParametersPrependTextCtrlEvtText)
        self.cudaminerParametersTextCtrl.Bind(wx.EVT_TEXT, self.onCudaminerParametersTextCtrlEvtText)
        self.cudaminerParametersPrepend = "optirun"
        self.cudaminerParameters = ""
        cudaminerParametersBoxWidget.Add(cudaminerParametersPrependLabel, pos=(0, 0), flag=wx.EXPAND)
        cudaminerParametersBoxWidget.Add(self.cudaminerParametersPrependTextCtrl, pos=(0, 1), flag=wx.EXPAND)
        cudaminerParametersBoxWidget.Add(cudaminerParametersLabelWidget, pos=(1, 0), flag=wx.EXPAND)
        cudaminerParametersBoxWidget.Add(self.cudaminerParametersTextCtrl, pos=(1, 1), flag=wx.EXPAND)
        cudaminerParametersBoxWidget.AddGrowableCol(1)
        cudaminerParametersCollapsiblePaneWidget.GetPane().SetSizer(cudaminerParametersBoxWidget)
        cudaminerParametersBoxWidget.SetSizeHints(cudaminerParametersCollapsiblePaneWidget.GetPane())
        if self.cudaminerParametersPrepend != "" or self.cudaminerParameters != "":
            cudaminerParametersCollapsiblePaneWidget.Expand()
        panelBoxWidget.Add(cudaminerParametersCollapsiblePaneWidget, 0, flag=wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.GROW, border=15)
        panelBoxWidget.Add(wx.StaticLine(self.panel, -1, style=wx.LI_HORIZONTAL), flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
        # value generation controls
        valueGenerationSizer = wx.GridBagSizer(hgap=5, vgap=5)
        numberOfValuesLabelWidget = wx.StaticText(self.panel, label="Number of hash/s values:" )
        self.hashRateCountSpinCtrl = wx.SpinCtrl(self.panel, value=str(self.hashRateCount), min=1)
        self.hashRateCountSpinCtrl.Bind(wx.EVT_SPINCTRL, self.onhashRateCountSpinCtrlEvtSpinctrl)
        numberOfCandidatesSpinCtrlLabel = wx.StaticText(self.panel, label="Number of result values (0 means unlimited):")
        self.numberOfCandidatesSpinCtrl = wx.SpinCtrl(self.panel, value=str(self.numberOfCandidates), min=0, max=1073741824)
        self.numberOfCandidatesSpinCtrl.Bind(wx.EVT_SPINCTRL, self.onNumberOfCandidatesSpinCtrlEvtSpinctrl)
        valueGenerationSizer.Add(numberOfValuesLabelWidget, pos=(0, 0), flag=wx.EXPAND)
        valueGenerationSizer.Add(self.hashRateCountSpinCtrl, pos=(0, 1), flag=wx.EXPAND)
        valueGenerationSizer.Add(numberOfCandidatesSpinCtrlLabel, pos=(1, 0), flag=wx.EXPAND)
        valueGenerationSizer.Add(self.numberOfCandidatesSpinCtrl, pos=(1, 1), flag=wx.EXPAND)
        panelBoxWidget.Add(valueGenerationSizer, flag=wx.EXPAND|wx.LEFT|wx.BOTTOM|wx.RIGHT, border=15)
        generationOptionsCollapsiblePane = wx.CollapsiblePane(self.panel, label="Value generation options:")
        generationOptionsCollapsiblePaneSizer = wx.GridBagSizer(hgap=5, vgap=5)
        outputScanIntervalLabel = wx.StaticText(generationOptionsCollapsiblePane.GetPane(), label="Output scan interval (seconds):")
        outputScanIntervalSpinCtrl = wx.SpinCtrl(generationOptionsCollapsiblePane.GetPane())
        generationOptionsCollapsiblePaneSizer.Add(outputScanIntervalLabel, pos=(0, 0), flag=wx.EXPAND)
        generationOptionsCollapsiblePaneSizer.Add(outputScanIntervalSpinCtrl, pos=(0, 1), flag=wx.EXPAND)
        outputScanMaxCountLabel = wx.StaticText(generationOptionsCollapsiblePane.GetPane(), label="Output scan max. count:")
        outputScanMaxCountSpinCtrl = wx.SpinCtrl(generationOptionsCollapsiblePane.GetPane())        
        generationOptionsCollapsiblePaneSizer.Add(outputScanMaxCountLabel, pos=(1, 0), flag=wx.EXPAND)
        generationOptionsCollapsiblePaneSizer.Add(outputScanMaxCountSpinCtrl, pos=(1, 1), flag=wx.EXPAND)
        generationOptionsCollapsiblePaneSizer.AddGrowableCol(1)
        generationOptionsCollapsiblePane.GetPane().SetSizer(generationOptionsCollapsiblePaneSizer)
        generationOptionsCollapsiblePaneSizer.SetSizeHints(generationOptionsCollapsiblePane.GetPane())
        panelBoxWidget.Add(generationOptionsCollapsiblePane, flag=wx.EXPAND|wx.LEFT|wx.BOTTOM|wx.RIGHT, border=15)
        panelBoxWidget.Add(wx.StaticLine(self.panel, -1, style=wx.LI_HORIZONTAL), flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=15)
        # program controls
        buttonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.startButton = wx.Button(self.panel, label="Start")
        self.clearButton = wx.Button(self.panel, label="Clear results")
            # clearing results should only work when generation is stopped
        self.cancelSaveButton = wx.Button(self.panel, label="Save intermediate results and cancel")
        self.cancelButton = wx.Button(self.panel, label="Cancel")
        buttonsSizer.Add(self.startButton, flag=wx.LEFT|wx.RIGHT|wx.TOP|wx.BOTTOM, border=15)
        buttonsSizer.Add(self.cancelSaveButton, flag=wx.RIGHT|wx.TOP|wx.BOTTOM, border=15)
        buttonsSizer.Add(self.cancelButton, flag=wx.RIGHT|wx.TOP|wx.BOTTOM, border=15)
        buttonsSizer.Add(self.clearButton, flag=wx.RIGHT|wx.TOP|wx.BOTTOM, border=15)
        panelBoxWidget.Add(buttonsSizer) 
        self.gaugeProgressText = wx.StaticText(self.panel)
        self.gaugeWidget = wx.Gauge(self.panel, range=100)
        panelBoxWidget.Add(self.gaugeProgressText, flag=wx.ALIGN_CENTER_VERTICAL|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
        panelBoxWidget.Add(self.gaugeWidget, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
        panelBoxWidget.Add(wx.StaticLine(self.panel, -1, style=wx.LI_HORIZONTAL), flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
        self.generationThread = None
        self.startButton.Bind(wx.EVT_BUTTON, self.onStart)
        self.clearButton.Bind(wx.EVT_BUTTON, self.onClear)
        self.cancelButton.Bind(wx.EVT_BUTTON, self.onCancel)
        self.cancelSaveButton.Bind(wx.EVT_BUTTON, self.onCancelSave)       
        # summary table
        self.summaryGridScrolledPanel = wx.lib.scrolledpanel.ScrolledPanel(self.panel, )#size=(425,400))
        summaryGridScrolledPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.summaryGrid = wx.grid.Grid(self.summaryGridScrolledPanel)
        self.summaryGrid.CreateGrid(0,3)
        self.summaryGrid.GetTable().SetColLabelValue(0, "hash/s mean")
        self.summaryGrid.GetTable().SetColLabelValue(1, "hash/s values")
        self.summaryGrid.GetTable().SetColLabelValue(2, "command arguments")
        self.summaryGrid.AutoSize()
        summaryGridScrolledPanelSizer.Add(self.summaryGrid,1,wx.ALL|wx.EXPAND,5)
        self.summaryGridScrolledPanel.SetSizer(summaryGridScrolledPanelSizer)
        self.summaryGridScrolledPanel.Layout()
        self.summaryGridScrolledPanel.SetupScrolling()
        panelBoxWidget.Add(self.summaryGridScrolledPanel, 1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
        # menubar
        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()
        fileMenuStartItem = fileMenu.Append(-1, "&Start")
        self.Bind(wx.EVT_MENU, self.onStart, fileMenuStartItem)
        fileMenuClearItem = fileMenu.Append(-1, "&Clear")
        self.Bind(wx.EVT_MENU, self.onClear, fileMenuClearItem)
        fileMenuResumeItem = fileMenu.Append(-1, "&Resume")
        self.Bind(wx.EVT_MENU, self.onResume, fileMenuResumeItem)
        fileMenuCancelItem = fileMenu.Append(-1, "&Cancel")
        self.Bind(wx.EVT_MENU, self.onCancel, fileMenuCancelItem)
        fileMenuCancelSaveItem = fileMenu.Append(-1, "&Save intermediate results and cancel")
        self.Bind(wx.EVT_MENU, self.onCancelSave, fileMenuCancelSaveItem)
        fileMenu.AppendSeparator()
        fileMenuExitItem = fileMenu.Append(-1, text='&Exit')
        self.Bind(wx.EVT_MENU, self.onExit, fileMenuExitItem)
        helpMenu = wx.Menu()
        helpMenuAboutItem = helpMenu.Append(-1, text='&About')
        self.Bind(wx.EVT_MENU, self.onAboutBox, helpMenuAboutItem)
        menuBar.Append(fileMenu, '&File')
        menuBar.Append(helpMenu, '&Help')
        self.SetMenuBar(menuBar)  
        # init finalization
        self.generationWidgetsEnabled = set([self.cancelButton, self.cancelSaveButton, fileMenuCancelItem, fileMenuCancelSaveItem, ])
            # a collection which allows easier control of enabling and disabling controls
        self.generationWidgetsDisabled = set([self.startButton, self.clearButton, fileMenuStartItem, fileMenuClearItem, fileMenuResumeItem, ])
        if self.generationWidgetsEnabled & self.generationWidgetsDisabled != set(): # testing for `!= {}` doesn't work
            raise Exception("Implementation error: sets of widgets to be disabled and enabled during generation have to be disjoint")
        self.panel.SetSizer(panelBoxWidget)
        frameSizer.Add(self.panel, 1, wx.GROW|wx.ALL)
        self.SetSizer(frameSizer)
        self.__handle_controls_generation_stopped__()
        self.resultDictKeysSortedLast = []
    
    def onNumberOfCandidatesSpinCtrlEvtSpinctrl(self, event):
        self.numberOfCandidates = self.numberOfCandidatesSpinCtrl.GetValue()
    
    def __clear__(self, storage_file_path=__storage_file_path_default__()):
        self.summaryGrid.ClearGrid()
        self.generator = CudaminerParamCheckerGenerator(storage_file_path=storage_file_path)
    
    def onClear(self, event):
        self.__clear__()
    
    def onhashRateCountSpinCtrlEvtSpinctrl(self, event):
        self.hashRateCount= self.hashRateCountSpinCtrl.GetValue()
    
    def onCudaminerParametersPrependTextCtrlEvtText(self, event):
        self.cudaminerParametersPrepend = self.cudaminerParametersPrependTextCtrl.GetValue()
    
    def onCudaminerParametersTextCtrlEvtText(self, event):
        self.cudaminerParameters = self.cudaminerParametersTextCtrl.GetValue()
    
    def onResume(self, e):
        if self.generator.isRunning():
            answer = wx.MessageBox("Do you want to cancel the current value generation?", 'Info', wx.YES|wx.NO | wx.ICON_INFORMATION)
            if answer == wx.NO:
                return
            self.onCancel()
            answer = wx.MessageBox("Do you want to save the intermediate results?", 'Info', wx.YES|wx.NO | wx.ICON_INFORMATION)
            if answer == wx.YES:
                self.onCancelSave()
        fileDialogWidget = wx.FileDialog(self)
        fileDialogWidget = wx.FileDialog(self, "Open file with intermediate results - %s (%s)" % (cudaminer_param_checker_globals.app_name, cudaminer_param_checker_globals.app_version_string), "", "",
                                       "XYZ files (*.xyz)|*.xyz", wx.FD_OPEN)
        fileDialogWidgetResult = fileDialogWidget.ShowModal()
        if fileDialogWidgetResult == wx.ID_CANCEL:
            return
        fileDialogWidgetPath = fileDialogWidget.GetPath() # returns a unicode 
            # path
        filePath = str(fileDialogWidgetPath)
        self.__clear__(storage_file_path=filePath) # creates a new generator instance
        self.__summary_grid_update__(self.generator.getResultDict())
        logger.info("resuming with '%d' entries from file '%s'" % (len(self.generator.getResultDict()), filePath))
        wx.CallAfter(self.gaugeProgressText.SetLabel, "Generated %d/%d values" % (self.generator.getProgressCurrent(), self.generator.getProgressMax()))
        self.__start__()
    
    def onExit(self, e):
        self.onCloseWindow(e)

    def onAboutBox(self, e):
        description = """Description here"""
        licence = """This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Dieses Programm ist Freie Software: Sie können es unter den Bedingungen
der GNU General Public License, wie von der Free Software Foundation,
Version 3 der Lizenz oder (nach Ihrer Wahl) jeder neueren
veröffentlichten Version, weiterverbreiten und/oder modifizieren.

Dieses Programm wird in der Hoffnung, dass es nützlich sein wird, aber
OHNE JEDE GEWÄHRLEISTUNG, bereitgestellt; sogar ohne die implizite
Gewährleistung der MARKTFÄHIGKEIT oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.
Siehe die GNU General Public License für weitere Details.

Sie sollten eine Kopie der GNU General Public License zusammen mit diesem
Programm erhalten haben. Wenn nicht, siehe <http://www.gnu.org/licenses/>."""
        info = wx.AboutDialogInfo()
        #info.SetIcon(wx.Icon('.png', wx.BITMAP_TYPE_PNG))
        info.SetName(cudaminer_param_checker_globals.app_name)
        info.SetVersion(cudaminer_param_checker_globals.app_version_string)
        info.SetDescription(description)
        info.SetCopyright('(C) %s %s' % (cudaminer_param_checker_globals.app_copyright_year, cudaminer_param_checker_globals.app_author, ))
        info.SetWebSite(cudaminer_param_checker_globals.app_website)
        info.SetLicence(licence.decode("utf-8"))
        # adding developer etc. opens a credit section and it's strange to 
        # thank your self exclusively
        #info.AddDeveloper(cudaminer_param_checker_globals.app_author)
        #info.AddDocWriter(cudaminer_param_checker_globals.app_author)
        #info.AddArtist()
        #info.AddTranslator()
        wx.AboutBox(info)
        
    def __check_generation_running_callback__(self):
        return self.generator.isRunning()
    
    def __handle_controls_generation_running__(self):
        for widget in self.generationWidgetsEnabled:
            widget.Enable()
        for widget in self.generationWidgetsDisabled:
            widget.Enable(enable=False) 
                # using `Enable(enable=False)` instead of `Disable()` makes the 
                # function usable for `wx.Button` and `wx.MenuItem`
    
    def __handle_controls_generation_stopped__(self):
        for widget in self.generationWidgetsEnabled:
            widget.Enable(enable=False)
        for widget in self.generationWidgetsDisabled:
            widget.Enable()
                # using `Enable(enable=False)` instead of `Disable()` makes the 
                # function usable for `wx.Button` and `wx.MenuItem`
    
    def __start__(self):
        # code to be reused in `onStart` and `onResume` (allows individual progress label handling and avoids passing event=None to an actual event handling function, e.g. onStart)
        self.generator.start()
        def __update_callback__(value_count_current, value_count_max):
            progress = value_count_current/value_count_max
            wx.CallAfter(self.gaugeWidget.SetValue, progress*1000)
            wx.CallAfter(self.gaugeProgressText.SetLabel, "Generated %s/%s values" % (str(value_count_current), str(value_count_max)))
            if not self.generator.isRunning():
                return
            summaryGridRowCount = self.summaryGrid.GetTable().GetRowsCount()
            wx.CallAfter(self.__summary_grid_update__, self.generator.getResultDict())
        def __thread__():
            if self.cudaminerParameters == "":
                cudaminerParameters = []
            else:
                cudaminerParameters = self.cudaminerParameters.split(" ")
            if self.cudaminerParametersPrepend == "":
                cudaminerParametersPrepend = []
            else:
                cudaminerParametersPrepend = self.cudaminerParametersPrepend.split(" ")
            self.generator.start()
            self.generator.generate_cudaminer_param_checker_values(hash_rate_count=self.hashRateCount, update_callback=__update_callback__, check_running_callback=self.__check_generation_running_callback__, cudaminer_parameters_prepend=cudaminerParametersPrepend, cudaminer_additional_parameters=cudaminerParameters, )
            self.__handle_controls_generation_stopped__()
        generationThread = threading.Thread(target=__thread__)
        generationThread.start()
        self.__handle_controls_generation_running__()
    
    def onStart(self, event):
        wx.CallAfter(self.gaugeProgressText.SetLabel, "Generated 0/%d values" % (self.generator.getProgressMax(), )) # avoid that the old value is still displayed between clicking start and the first update (when resuming this will be overwritten immediately with and update)
        self.__start__()
    
    def __summary_grid_update__(self, resultDict):
        """in order to update the table as efficient as possible, the new `resultDictKeysSorted` is compared to `self.resultDictKeysSortedLast` and the difference (always one item) inserted into the table. Calculating the intersetion of `set`s and sorting of already sorted collections is very efficient
        @TODO: research how efficiently `wx.Grid` updates its view (e.g. whether it skips updates which are not in the viewport)"""
        resultDictKeysSorted = sorted(resultDict.items(), key=lambda x:x[1][0])
        resultDictKeysSortedIntersect = set(resultDictKeysSorted)-(set(resultDictKeysSorted) & set(self.resultDictKeysSortedLast))
        newItems = sorted(list(resultDictKeysSortedIntersect), key=lambda x:x[1][0]) # sorting allows the function to work for both single value and initial setup after resumption because `wx.grid.Grid.InsertRows` below never exceeds the row count
        logger.debug("updating grid with '%d' new items" % (len(newItems), ))
        for newItem in newItems:
            newItemIndex = resultDictKeysSorted.index(newItem)
            logger.debug("inserting row into grid at index '%d'" % (newItemIndex, ))
            self.summaryGrid.InsertRows(pos=newItemIndex)
            key = __unmarshal_shelve_key__(resultDictKeysSorted[newItemIndex][0])
            value = resultDictKeysSorted[newItemIndex][1]
            self.summaryGrid.GetTable().SetValue(row=newItemIndex, col=0, value=str(round(value[0], 2)))
            self.summaryGrid.GetTable().SetValue(row=newItemIndex, col=1, value=str(value[1]))
            self.summaryGrid.GetTable().SetValue(row=newItemIndex, col=2, value=str(key))
        self.resultDictKeysSortedLast = resultDictKeysSorted
    
    def onCancel(self, event):
        self.generator.stop()
        self.__handle_controls_generation_stopped__()
    
    def onCancelSave(self, event):
        self.generator.stop()
        self.saveIntermediateResult()
        self.__handle_controls_generation_stopped__()

    def onCloseWindow(self, event):
        self.generator.stop()
        if not self.generationThread is None:
            generationThread.join() # avoid PyDeadObjectError which seems to 
                # occur after `Destroy` has been called. This cause long 
                # shutdown (up to 90 seconds depending on cudaminer) -> it's more elegant to expose the cudaminer process and kill it immediately because there's no sense in awaiting its return when the application ought to be closed
        self.Destroy()
    
    def saveIntermediateResult(self):
        fileDialogWidget = wx.FileDialog(self.panel)
        fileDialogWidget = wx.FileDialog(self, "Save intermediate results to file - %s (%s)" % (cudaminer_param_checker_globals.app_name, cudaminer_param_checker_globals.app_version_string), "", "",
                                       "XYZ files (*.xyz)|*.xyz", wx.FD_OPEN)
        fileDialogWidgetResult = fileDialogWidget.ShowModal()
        confirmCancel = False
        while fileDialogWidgetResult == wx.ID_CANCEL:
            answer = wx.MessageBox("Do you really want to cancel saving the result (will be lost)?", 'Info', wx.YES|wx.NO | wx.ICON_INFORMATION)
            confirmCancel = answer == wx.YES
            if not confirmCancel:
                fileDialogWidgetResult = fileDialogWidget.ShowModal()
            return
        fileDialogWidgetPath = fileDialogWidget.GetPath() # returns a unicode 
            # path
        filePath = str(fileDialogWidgetPath)
        result_dict_target = shelve.open(filePath)
        for k,v in self.generator.getResultDict().items():
            result_dict_target[k] = v
        result_dict_target.close()

@plac.annotations(
    frontend=("the frontend to be used to interact with the generation program", 'positional', None, str, frontends), 
    cudaminer=(__cudaminer_docstring__, 'positional', None, str), 
    output_scan_interval=(__output_scan_interval_docstring__, 'positional', None, int, ), 
    output_scan_max_count=(__output_scan_max_count_docstring__, 'positional', None, int),
    hash_rate_count=(__hash_rate_count_docstring__, 'positional', None, int, ), 
    debug=("Turn on debugging output", "flag")
)
def cudaminer_param_checker(frontend=frontend_default, cudaminer=cudaminer_default, output_scan_interval=output_scan_interval_default, output_scan_max_count=output_scan_max_count_default, hash_rate_count=hash_rate_count_default, debug=debug_default):
    """
    @args cudaminer %(__cudaminer_docstring__)s
    @args output_scan_interval %(__output_scan_interval_docstring__)s
    @args output_scan_max_count %(__output_scan_max_count_docstring__)s
    @args hash_rate_count %(__hash_rate_count_docstring__)s""" % {"__cudaminer_docstring__": __cudaminer_docstring__, "__output_scan_interval_docstring__": __output_scan_interval_docstring__, "__output_scan_interval_docstring__": __output_scan_interval_docstring__, "__output_scan_max_count_docstring__": __output_scan_max_count_docstring__, "__hash_rate_count_docstring__": __hash_rate_count_docstring__}
    if debug == True:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    if frontend == FRONTEND_CONSOLE_GUI:
        visualize_cudaminer_param_checker_results_console_gui()
    elif frontend == FRONTEND_WXPYTHON_GUI:
        visualize_cudaminer_param_checker_results_wxpython_gui()
    else:
        raise ValueError("frontend has to be one of '%s', but is '%s'" % (str(frontends), frontend))

if __name__ == "__main__":
    plac.call(cudaminer_param_checker)

