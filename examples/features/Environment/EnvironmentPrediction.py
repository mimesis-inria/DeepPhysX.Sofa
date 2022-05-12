"""
EnvironmentPrediction.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
SofaEnvironment compatible with SOFA GUI and Prediction pipeline.
Apply predictions.
"""

# Python related imports
import os
import sys
from numpy import array

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnvironmentSofa import MeanEnvironmentSofa


# Create an Environment as a BaseEnvironment child class
class MeanEnvironmentPrediction(MeanEnvironmentSofa):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        MeanEnvironmentSofa.__init__(self,
                                     root_node=root_node,
                                     ip_address=ip_address,
                                     port=port,
                                     instance_id=instance_id,
                                     number_of_instances=number_of_instances,
                                     as_tcp_ip_client=as_tcp_ip_client,
                                     environment_manager=environment_manager)

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
        - recv_parameters
        - create
    """

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - apply_prediction
    """

    def apply_prediction(self, prediction):
        # Update MechanicalObject
        self.MO['predict'].position.value = array([prediction])
