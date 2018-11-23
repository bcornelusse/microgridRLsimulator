============================
What is microgridRLsimulator
============================


A microgrid is a small power system connecting devices that consume, generate and store electricity. 
Usually microgrids are able to operate in islanded mode (off-grid), but they can also be connected to the public grid. 
We are interested mostly in the latter case, because it offers many more valorization mechanisms.

microgridRLsimulator is a python tool that aims at simulating the techno-economics of a microgrid,
and in particular at quantifying the performance of an agent responsible for controlling the devices of the microgrids as a function
of the random processes governing all the variables that impact the microgrid operation 
(e.g. consumption, renewable generation, market prices).

It offers the following functionalities:

* To simulate a control policy on real data
* New datasets can be easily integrated
* The microgrid topology can be easily configured
* Results are stored in the ``results`` folder
* Plots are automatically generated and can be regenerated from a set of existing results
