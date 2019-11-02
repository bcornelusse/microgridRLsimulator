0.2
===

The microgrid is off-grid and the goal is to minimize the exploitation cost.
The agent only has access to the current non-steerable (i.e. renewable) generation and non-flexible consumption in the microgrid. It has also access to the state of charge of the different storages and it must decide how to use the storage systems.
The genset, i.e. the diesel steerable generation, compensates to establish the equilibrium. In case there is an excess of non-steerable generation and no more room for storage,
the non-steerable generation is "curtailed", i.e. is lost.

Components considered
---------------------
So far, the following devices are available:

* Non-flexible loads:
    Loads that need to be served. Defined in the microgrid configuration ``json`` file.

* Non-flexible generation:
    Renewable production that is not steerable. Defined in the microgrid configuration ``json`` file.

* Storage:
    There are two options for considering a storage components in the microgrid. i) A simple battery model that assumes limited capacity, max (dis)charge rates and(dis)charge efficiencies.
    The parameters are defined in the microgrid configuration ``json`` file. ii )A decreasing capacity battery model where the
    available capacity is degrading as a function of its usage (i.e. the total number of cycles performed).


* Steerable generators:
    When the energy  level from storages and from non-flexible production is not sufficient to ensure the loads are served, the steerable generators compensate for the remaining energy to be supplied.
    In order to keep a high production efficiency, the steerable generators cannot works at very low power. Hence, a minimum stable generation is specified for each steerable generator. The parameters 
    are defined in the microgrid configuration ``json`` file.
    

State representation
--------------------

The available information at each time-step is composed of the consumption, the state of charge of each storage device, the renewable production and the current date time. The date time is represented by the number of hours spent since January 1 at midnight of the current year.
The state is provided as a list of the form::

    [ non_flex_consumption , [state_of_charge_0, state_of_charge_1,...] , non_flex_production, date_time]

Action representation
---------------------

We assume that the agent has control of the storage devices. However, the original action space is continuous and of high-dimensionality.
High-level actions are used in the decision making process that are then mapped into the original action space.

The high-level actions available at each decision step are the charging (C), discharging (D) and idling (I) of each storage device in the micro-grid.
For more than one storage the product of all these actions is comprising the action space. However the simultaneous charging of one and discharging of
another storage is considered impossible and thus removed from the action space.

High level actions are then converted in implementable actions automatically following a rule-based strategy:
 1. If the total possible production (i.e. RES production, active steerable generators capacity and storages maximum discharge rate) is lower than the total consumption, a steerable generator is activated at its minimum stable generation. This instruction is repeated until the total load can be served or until all steerable generators are active. In a few words, the generators are activated one by one at their minimum stable generation until the total load can be served.
 2. Once all active steerable generators are known, the net generation can be calculated based on their minimum stable generation, the RES production and the total consumption.
 3. If the net generation is positive, the storages (with charge instruction) charges the excess of energy until the net generation becomes zero. The storages with discharge or idle instructions don't do anything. The remaining excess of energy is curtailed.
 4. If the net generation is negative, the storages (with discharge instruction) discharges the deficit of energy until the net generation becomes zero. The storages with charge or idle instructions don't do anything. The remaining deficit of energy is then compensated by the active steerable generators which can be ajusted at a higher production level than their minimimum stable power. If, in addtion, steerable generators can't handle the remaining deficit, this deficit is considered as lost load. 


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

Four agents are already implemented.

IdleAgent
~~~~~~~~~
This is a template of agent that takes no action.

DQNAgent
~~~~~~~~
This agent uses deep Q-Learning.

RandomAgent
~~~~~~~~~~~
This agent takes random actions from the action space.

HeuristicAgent
~~~~~~~~~~~~~~
This agent charges when the renewable production is higher than the consumption and it discharges otherwise.

OptimizationAgent
~~~~~~~~~~~~~~~~~
This agent solves a linear program that minimizes the cost to optimize its actions. The output actions are low level, that means continuous actions showing the exact charge/discharge level of each storage and the exact generation from steerable generators. Two parameters can be adjusted: the control horizon (the number of lookahead periods in the optimization problem) 
and the simulation horizon (the number of actions steps that will be simulated). The lookahead works with a forecast. For now, the forecast is the exact data. Another parameter ``save_data`` is used to allow the user to save the state-action pairs found by the optimization controller in datasets in order to train a supervised learning model used in SLAgent.

SLAgent
~~~~~~~
As mentionned previously, the SLAgent (supervised learning agent) is an agent that maps states to actions by learning a model. The dataset used for learning can be created with the OptimizationAgent by setting ``save_data`` parameter to ``True``. Since, the data comes from the optimization controller, the actions of this agent are low-level actions (continuous actions). To load the dataset, you have to specify the parameter 
``control_horizon_data`` to be equal to the control horizon used in the dataset you want to load.
The control horizon used to make a dataset can be found in its name. For example, ``elespino_actions_12.txt`` and ``elespino_states_12.txt`` datasets are made using a control_horizon of 12. The learning algorithm used is [still to be definied]. Do not forget to use this agent only on unseen data. For example, if the learning dataset takes data from year 2016 of elespino, you can only test this agent in the year 2017.

Microgrid Configuration
-----------------------

The microgrid configuration is described in a JSON file. It consists of a description of the devices used in the microgrid as well as some additional information such as costs, simulation time step and objectives.

Devices
~~~~~~~
+--------------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
| Load                     | name                   | The load name.                                                                                                                                                                                                                                                                                                                                                                                                                                              | 
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | capacity               | The load capacity [kW].                                                                                                                                                                                                                                                                                                                                                                                                                                     | 
+--------------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
| Generator                | name                   | The generator name.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | capacity               | The generator capacity [kW].                                                                                                                                                                                                                                                                                                                                                                                                                                |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | steerable              | Flag that indicates if the generator is steerable (i.e. non renewable) or not.                                                                                                                                                                                                                                                                                                                                                                              |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | operating_point        | Indicates the decreasing rate of the capacity of a renewable energy source over time [days][-]. For example, ``[365, 0.96]`` means the capacity decreases by 4% per year. This element must not be used if steerable is set to ``True``.                                                                                                                                                                                                                    |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | min_stable_generation  | The minimum power ratio the steerable generator must provide when it is active [-]. The ratio is related to the generator capacity. This element must not be used if steerable is set to ``False``.                                                                                                                                                                                                                                                         |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | diesel_price           | The price to produce 1 L of diesel [€/L]. This element must not be used if steerable is set to ``False``.                                                                                                                                                                                                                                                                                                                                                   |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | operating_point_1      | Indicates the first operating point of the steerable generator on the fuel curve [kW][l/h] . This element must not be used if steerable is set to ``False``.                                                                                                                                                                                                                                                                                                |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | operating_point_2      | Indicates the second operating point of the steerable generator on the fuel curve [kW][l/h] . This element must not be used if steerable is set to ``False``.                                                                                                                                                                                                                                                                                               |
+--------------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Storage                  | name                   | The storage name.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | type                   | The storage type. The type can be set to either ``"Storage"`` or ``"DCAStorage"``. ``"Storage"`` is used to simulate a basic storage without any special feature. ``"DCAStorage"`` is used to simulate a decreasing capacity storage. The capacity of the DCAStorage is assumed to decrease linearly with the number of cycles. When DCAStorage is used decreasing rate must be specified in operating_point element.                                       |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | capacity               | The storage capacity [kWh]. For a decreasing capacity storage, this represents the initial capacity.                                                                                                                                                                                                                                                                                                                                                        |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | max_charge_rate        | The maximum charge rate of the storage [kW].                                                                                                                                                                                                                                                                                                                                                                                                                |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | min_charge_rate        | The minimum charge rate of the storage [kW].                                                                                                                                                                                                                                                                                                                                                                                                                |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | charge_efficiency      | The charge efficiency [-].                                                                                                                                                                                                                                                                                                                                                                                                                                  |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | discharge_efficiency   | The discharge efficiency [-].                                                                                                                                                                                                                                                                                                                                                                                                                               |
+                          +------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                          | operating_point        | Indicates the decreasing rate of the DCA Storage capacity in relation to the number of cycles [-][-]. For example, ``[3000, 0.7]`` means the capacity decreases by 30% when the number of cycles is 3000. This element must be specified only if the storage type is set to ``"DCAStorage"``.                                                                                                                                                               |
+--------------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Objectives & Others
~~~~~~~~~~~~~~~~~~~

The system is intended to become multi-objective. It could has to minimize the operation cost while ensuring the reliability by maximizing service level or served demand.
Hence, the reward must be tuned with respect to the desired objectives. For this version, don't change the values of the objectives because the reward is still ``-total_cost``. 

+--------------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
| Objectives               | total_cost             | Flag that indicates if the total cost is part of the reward information.                                                                     | 
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | fuel_cost              | Flag that indicates if the fuel (used by the steerable generators) cost is part of the reward information.                                   | 
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | load_shedding          | Flag that indicates if the lost load (quantity of non served load) is part of the reward information.                                        |
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | curtailment            | Flag that indicates if the amount is part of the reward information.                                                                         |
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | storage_maintenance    | Flag that indicates if the storage maintenance cost is part of the reward information.                                                       |
+--------------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+

For now, since the system is off grid we assume that, imports are equivalent to load shedding and exports are equivalent to production curtailment.

+--------------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
| Additional information   | curtailment_price      | The price [€/kWh] to pay to curtail 1 kWh.                                                                                                   | 
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | load_shedding_price    | The price [€/kWh] for each non served kWh.                                                                                                   | 
+                          +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+ 
|                          | period_duration        | The simulation timestep [min]. Must be a multiple of the dataset timestep.                                                                   |
+--------------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+

Data
----

An example is specified by two files located in the ``data`` folder. These two files should start with the name of the case.
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

Time-series from the ``case1_dataset.csv`` are used to simulate the 3 loads ``C1,C2,C3`` and the PV module ``EPV``. The technical limits of the generator i.e. the maximum (capacity)
and the minimum stable (percentage of the capacity) operating point are also specified. The storage devices have slightly different characteristics, namely different charging/discharging efficiencies.

In ``case1_dataset.csv`` data covers 2 years (2014 and 2015) with a frequency of 1 hour. There is a safeguard if you run a case out of this time range.


El Espino
~~~~~~~~~

El Espino is a real case of microgrid which is located in Bolivia. The microgrid contains 1 load, a PV module, a diesel generator and a storage device. This microgrid configuration data 
``elespino.json`` contains also costs related to curtailment and lost load.

Time-series from the ``elespino_dataset.csv`` are used to simulate the load C1 and the PV module EPV. The technical limits of the generator i.e. the maximum (capacity) and the minimum stable 
(percentage of the capacity) operating point are also specified. The storage devices have slightly different characteristics, namely different charging/discharging efficiencies.

The data covers the period from 2016-01-01 to 2017-07-31 with a frequency of 5 minutes. There is a safeguard if you run a case out of this time range.