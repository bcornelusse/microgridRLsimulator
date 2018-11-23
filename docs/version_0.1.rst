0.1
===

The microgrid is off-grid and the goal is to minimize the exploitation cost.
The agent only has access to the current non-steerable (i.e. renewable) generation and non-flexible consumption in the micrgrogrid, and must decide how to use the storage systems.
The genset, i.e. the diesel steerable generation, compensates to establish the equilibrium. In case there is an excess of non-steerable generation and no more room for storage,
the non-steerable generation is "curtailed", i.e. is lost.


So far, the following devices are available:

* Non-flexible loads
* Non-flexible generation
* Simple battery model: limited capacity, max (dis)charge rates, (dis)charge efficiencies


State representation
--------------------

The available information at each time-step is composed of the consumption, the state of charge of each storage device and the renewable production.
The state is provided as a list of the form::

    [ non_flex_consumption , [state_of_charge_0, state_of_charge_1,...] , non_flex_production]

Action representation
---------------------

We assume that the agent has control of the storage devices. However, the original action space is continuous and of high-dimensionality.
High-level actions are used in the decision making process that are then mapped into the original action space.

The high-level actions available at each decision step are the charging (C), discharging (D) and idling (I) of each storage device in the micro-grid.
For more than one storage the product of all these actions is comprising the action space. However the simultaneous charging of one and discharging of
another storage is considered impossible and thus removed from the action space.

High level actions are then converted in implementable actions automatically following a rule-based strategy.

Rewards
-------

The instantaneous reward is defined as the total cost of operation of the micro-grid and is composed of:

 a) fuel costs for the generation,
 b) curtailment cost for the excess of generation that had to be curtailed,
 c) load shedding cost for the excess of load that had to be shed in order to maintain balance in the grid.


Agents
------
An agent can be implemented by inheriting from the ``Agent`` class and implementing the required methods.

There are two ways to run a newly created agent:

 1. store your agent code where you want and use the ``--agent-file <new_agent>.py`` option to specify your python file as a input when you call the simulator,
 2. embed your new agent in the Agent submodule, make the necessary modification in ``__main__.py`` so as to make your agent accessible through the ``--agent <name>`` option.

Option 2 is more complex and we will use it to incorporate stable agent codes when issuing new versions. Hence in development, **option 1 is preferred**.

Two agents are already implemented.

IdleAgent
~~~~~~~~~
This is a template of agent that takes no action.

DQNAgent
~~~~~~~~
This agent uses deep Q-Learning.

Examples
--------

An example is specified by two files located in the ``examples/data`` folder. These two files should start with the name of the case.
For instance, for case 1, it should be ``case1.json`` and ``case1_dataset_csv``.


The JSON file is used to define the configuration of the micro-grid i.e. the components comprising the micro-grid and the technical specifications
of each component.

The CSV file contains time-series for the components defined in the JSON file (e.g. renewable production, consumption etc.). Each column corresponds to the
components defined in the JSON file and the name of the column should be identical to the name of the component. Dates should be in "yyyy-mmm-d HH:MM:SS" format,
fields separated by columns (;) and '.' used as a decimal separator.


case1
~~~~~


Case 1 is used as an example to illustrate the functionality of the micro-grid simulator. In the ``case1.json`` file the micro-grid configuration contains 3 loads, a PV module,
a diesel generator and 2 storage devices. Additionally the costs for curtailment and load shedding are defined.

Time-series from the ``case1_dataset_csv`` are used to simulate the 3 loads ``C1,C2,C3`` and the PV module ``EPV``. The technical limits of the generator i.e. the maximum (capacity)
and the minimum stable (percentage of the capacity) operating point are also specified. The storage devices have slightly different characteristics, namely different charging/discharging efficiencies.

In ``case1_dataset_csv`` data covers 2 years (2014 and 2015). There is no safeguard if you run a case out of this time range for now, the application will fail at some point.