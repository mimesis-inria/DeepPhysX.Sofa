"""
EnvironmentSofa.py
At each step, generate a random vector of 50 values between [0, pi] and compute its mean.
The random vector is the input of the network and the ground truth is the mean.
SofaEnvironment compatible with SOFA GUI.
Create scene graph and define behavior.
"""

# Python related imports
from numpy import mean, pi, array
from numpy.random import random, randint
from time import sleep

# DeepPhysX related imports
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


# Create an Environment as a BaseEnvironment child class
class MeanEnvironmentSofa(SofaEnvironment):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        SofaEnvironment.__init__(self,
                                 root_node=root_node,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

        # MechanicalObjects container
        self.MO = {}

        # Environment parameters
        self.constant = False
        self.data_size = [30, 3]
        self.sleep = False

    """
    ENVIRONMENT INITIALIZATION
    Methods will be automatically called in this order to create and initialize Environment.
        - recv_parameters
        - create
    """

    def recv_parameters(self, param_dict):
        # If True, the same data is always sent so one can observe how the prediction crawl toward the ground truth.
        self.constant = param_dict['constant'] if 'constant' in param_dict else self.constant
        # Define the data size
        self.data_size = param_dict['data_size'] if 'data_size' in param_dict else self.data_size
        # If True, step will sleep a random time to simulate longer processes
        self.sleep = param_dict['sleep'] if 'sleep' in param_dict else self.sleep

    def create(self):
        # Initialize data
        input_value = pi * random(self.data_size)
        output_value = array([mean(input_value, axis=0)])

        # Add SOFA plugins
        plugins = ['SofaComponentAll']
        self.root.addObject('RequiredPlugin', pluginName=plugins)
        # Add a MechanicalObject for the input data
        self.root.addChild('input')
        self.MO['input'] = self.root.input.addObject('MechanicalObject', name='MO', position=input_value.tolist(),
                                                     showObject=True, showObjectScale=5)
        # Add a MechanicalObject for the ground truth
        self.root.addChild('output')
        self.MO['output'] = self.root.output.addObject('MechanicalObject', name='MO', position=output_value.tolist(),
                                                       showObject=True, showObjectScale=10, showColor="0 200 0 255")
        # Add a MechanicalObject for the prediction
        self.root.addChild('predict')
        self.MO['predict'] = self.root.predict.addObject('MechanicalObject', name='MO', position=[0., 0., 0.],
                                                         showObject=True, showObjectScale=10, showColor="100 0 200 255")

    """
    ENVIRONMENT BEHAVIOR
    Methods will be automatically called by requests from Server or from EnvironmentManager depending on 
    'as_tcp_ip_client' configuration value.
        - onAnimateBeginEvent
        - onAnimateEndEvent
        - on_step
    """

    def onAnimateBeginEvent(self, event):
        # Compute new data
        if not self.constant:
            input_value = pi * random(self.data_size)
            output_value = array([mean(input_value, axis=0)])
            self.set_training_data(input_array=input_value, output_array=output_value)
        # Simulate longer process
        if self.sleep:
            sleep(0.01 * randint(0, 10))

    def onAnimateEndEvent(self, event):
        # Update MechanicalObjects
        self.MO['input'].position.value = self.input
        self.MO['output'].position.value = self.output

    def close(self):
        # Shutdown message
        print("Bye!")
