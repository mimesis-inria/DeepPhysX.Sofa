Environment
===========

.. _environment.sofaenvironment:

SofaEnvironment
---------------

Base:
:py:class:`BaseEnvironment.BaseEnvironment`

.. autoclass:: SofaEnvironment.SofaEnvironment
    :members: recv_parameters, create, init, send_parameters, send_visualization, step, check_sample, apply_prediction,
              close, set_training_data, set_loss_data, set_additional_dataset, reset_additional_datasets,
              get_prediction, update_visualisation

.. _environment.sofaenvironmentconfig:

SofaEnvironmentConfig
---------------------

Base:
:py:class:`BaseEnvironmentConfig.BaseEnvironmentConfig`

.. autoclass:: SofaEnvironmentConfig.SofaEnvironmentConfig
    :members: create_server, start_server, start_client, create_environment
