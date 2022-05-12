"""
EnvironmentDataset.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
SofaEnvironment compatible with DataGeneration pipeline.
Initialize and update visualization data.
"""

# Python related imports
import os
import sys

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from EnvironmentSofa import MeanEnvironmentSofa


# Create an Environment as a BaseEnvironment child class
class MeanEnvironmentDataset(MeanEnvironmentSofa):

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
        - send_visualization
    """

    def send_visualization(self):
        # Point cloud (object will have id = 0)
        self.factory.add_object(object_type="Points",
                                data_dict={"positions": self.MO['input'].position.value,
                                           "c": "blue",
                                           "at": self.instance_id,
                                           "r": 5})
        # Ground truth value (object will have id = 1)
        self.factory.add_object(object_type="Points",
                                data_dict={"positions": self.MO['output'].position.value,
                                           "c": "green",
                                           "at": self.instance_id,
                                           "r": 10})
        # Return the visualization data
        return self.factory.objects_dict

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
    """

    async def on_step(self):
        # Update visualization with new input and ground truth
        self.factory.update_object_dict(object_id=0,
                                        new_data_dict={'positions': self.MO['input'].position.value})
        self.factory.update_object_dict(object_id=1,
                                        new_data_dict={'positions': self.MO['output'].position.value})
        # Send visualization data to update
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
