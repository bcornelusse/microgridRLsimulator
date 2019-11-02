# -*- coding: utf-8 -*-

"""Microgrid RL simulator - version 0.1
Microgrids simulator designed for RL

Usage:
    microgridRLsimulator [options] <case>

where
    <case> is the name of the JSON file describing the microgridRLsimulator itself, assumed to be in the examples/data folder. #TODO specify path?

Options:
    -h                          Display this help.
    -o PATH                     Output path.
    --train_from_date DATETIME  Start of training simulation datetime [Default: 2016-01-01T00:00:00].
    --train_to_date DATETIME    End of training simulation datetime [Default: 2016-01-02T00:00:00].
    --test_from_date DATETIME   Start of testing simulation datetime [Default: 2016-01-01T00:00:00].
    --test_to_date DATETIME     End of testing simulation datetime [Default: 2016-01-07T00:00:00].
    --agent AGENT               Type of agent to use for simulation (Idle, DQN, Random, Heuristic, Optimization, SL, DQN, MCTS) [Default: DQN].
    --agent_options OPTIONS     Path to Json with options for the selected agent [Default: agent_options.json].
    --agent_file AGENT_FILE     Path to a Python file of a custom agent. This parameter overrides --agent parameter when specified.
    --rolling_strategy FLAG     Flag that indicates if the rolling strategy is active. [Default: False].
    --log_level LEVEL           Set logging level (debug, info, warning, critical) [Default: info].
    --log PATH                  Dump the log into a file
"""
import os

from microgridRLsimulator.agent.MCTSAgent import MCTSAgent
from microgridRLsimulator.agent.PPOAgent import PPOAgent
from microgridRLsimulator.agent.IdleAgent import IdleAgent
from microgridRLsimulator.agent.RandomAgent import RandomAgent
from microgridRLsimulator.agent.HeuristicAgent import HeuristicAgent
from microgridRLsimulator.agent.OptimizationAgent import OptimizationAgent
from microgridRLsimulator.agent.SLAgent import SLAgent
from microgridRLsimulator.agent.DQNAgent import DQNAgent
from microgridRLsimulator.gym_wrapper.microgrid_env import MicrogridEnv
from docopt import docopt
import logging
from dateutil.parser import isoparse
from dateutil.relativedelta import *
import json
import importlib


AGENT_TYPES = {IdleAgent.name(): IdleAgent, DQNAgent.name(): DQNAgent, RandomAgent.name(): RandomAgent,
               HeuristicAgent.name(): HeuristicAgent, OptimizationAgent.name(): OptimizationAgent,
               SLAgent.name(): SLAgent, PPOAgent.name(): PPOAgent, MCTSAgent.name(): MCTSAgent}
DEFAULT_CONTROLLER = DQNAgent

if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)

    # Create results/ folder (if needed)
    results_folder = "results/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set logging level
    numeric_level = getattr(logging, args['--log_level'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args['--log_level'])
    logging.basicConfig(level=numeric_level, filename=args['--log'])

    # Create logger
    logger = logging.getLogger(__name__)  #: Logger.



    # Configure the agent (--agent case)
    if args['--agent_file'] is None:
        if isinstance(args['--agent'], str):
            try:
                agent_type = AGENT_TYPES[args['--agent']]
            except KeyError:
                logger.error('Controller "%s" switch to the default controller (%s).' % (
                    args['--agent'], DEFAULT_CONTROLLER.__qualname__))
                agent_type = DEFAULT_CONTROLLER

    # Configure the agent (--agent_file case)
    else:
        try:
            agent_mod = importlib.import_module(args['--agent_file'].rsplit('.', 1)[0].replace('/', '.'))
            agent_type = agent_mod.agent_type
        except:
            logger.error('Controller "%s" switch to the default controller (%s).' % (
                    args['--agent_file'], DEFAULT_CONTROLLER.__qualname__))
            agent_type = DEFAULT_CONTROLLER

    # Instantiate the agent
    if args['--agent_options'] is not None:
        agent_options_path = "agent_options/" + args['--agent'] + "/" + args['--agent_options']
        with open(agent_options_path, 'rb') as jsonFile:
            agent_options = json.load(jsonFile)[args['--agent']]
    else:
        agent_options = {}
    # Create the training  and the testing simulation environments
    case = args['<case>']
    train_start_date = isoparse(args['--train_from_date'])
    train_end_date = isoparse(args['--train_to_date'])
    test_start_date = isoparse(args['--test_from_date'])
    test_end_date = isoparse(args['--test_to_date'])
    if args['--rolling_strategy'] is None or args['--rolling_strategy'] == "False":
        TrainSimulationEnvironment = MicrogridEnv(train_start_date, train_end_date, case, "Train")
        TestSimulationEnvironment = MicrogridEnv(test_start_date, test_end_date, case, "Test")
        agent = agent_type(TrainSimulationEnvironment, **agent_options)

        # Run the experiment
        agent.train_agent()
        agent.simulate_agent()
        agent.set_environment(TestSimulationEnvironment)
        agent.simulate_agent()
    else:
        if "elespino" in case:
            train_start_date = isoparse("2016-01-01T00:00:00")
        elif "case1" in case:
            train_start_date = isoparse("2014-01-01T00:00:00")
        for n_training_months in range(2, 18): # number of training month increase / the start does not change
            train_end_date = train_start_date + relativedelta(months=+n_training_months)
            test_start_date = train_end_date
            test_end_date = test_start_date + relativedelta(months=+1) # test during the next month
            TrainSimulationEnvironment = MicrogridEnv(train_start_date, train_end_date, case, "Train")
            TestSimulationEnvironment = MicrogridEnv(test_start_date, test_end_date, case, "Test")
            agent = agent_type(TrainSimulationEnvironment, **agent_options)
            agent.train_agent()
            agent.simulate_agent(agent_options=agent_options)
            agent.set_environment(TestSimulationEnvironment)
            agent.simulate_agent(agent_options=agent_options)
            if n_training_months == 12:
                train_end_date = train_start_date + relativedelta(months=+n_training_months)
                test_start_date = train_end_date
                test_end_date = test_start_date + relativedelta(months=+6)
                TrainSimulationEnvironment = MicrogridEnv(train_start_date, train_end_date, case, "Train")
                TestSimulationEnvironment = MicrogridEnv(test_start_date, test_end_date, case, "Test")
                agent = agent_type(TrainSimulationEnvironment, **agent_options)
                agent.train_agent()
                agent.set_environment(TestSimulationEnvironment)
                agent.simulate_agent(agent_options=agent_options)

