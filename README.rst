Installation
============

Be sure you have python 3.6 installed.
Run ``install.sh``. This creates the python environment ``mgenv`` with the necessary requirements to run the application.

Running the application
=======================

Be sure you are using the python environment ``mgenv`` (``source menv/bin/activate``).


Get help::

    python -m microgridRLsimulator -h

Run the application::

    python -m microgridRLsimulator --from_date 20150101T00:00:00 --to_date 20150102T00:00:00 --agent DQN --agent_options agent_options.json case1

This runs the case1 examples located in folder ``examples/data`` using a DQNAgent, that is a agent that uses Deep Q learning.

Documentation
=============

You can generate this documentation yourself:

::

    cd <to the root of the project>
    cd docs; make html; cd ..

The html doc is in ``_build/html``

