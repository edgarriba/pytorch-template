# pytorch-template
Repository containing a template training project

Development Setup
=================

Assuming that you are on ubuntu 18.04, with nvidia-drivers installed.

In bash, source the ``path.bash.inc`` script.  This will install a
local conda environment under ``./.dev_env``, which includes pytorch
and some dependencies (no root required).

To install, or update the conda environment run ``setup_dev_env.sh``

.. code:: bash

    ./setup_dev_env.sh

How to run
==========

In order to launch the training script,

Activate the conda environment:

.. code:: bash

    source path.bash.inc
    
Make sure the conda environment is activated and run: 

.. code:: bash

    python main.py --input-dir ~/data_dir --output-dir ~/output --output-dir-val ~/output_val
