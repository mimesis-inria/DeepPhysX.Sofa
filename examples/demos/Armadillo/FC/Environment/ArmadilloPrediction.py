"""
ArmadilloPrediction
Simulation of an Armadillo with NN computed simulations.
The SOFA simulation contains the models used to apply the network predictions.
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys
import numpy as np

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ArmadilloTraining import ArmadilloTraining
from parameters import p_forces, p_model


class ArmadilloPrediction(ArmadilloTraining):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 visualizer=None):

        ArmadilloTraining.__init__(self,
                                   as_tcp_ip_client=as_tcp_ip_client,
                                   instance_id=instance_id,
                                   instance_nb=instance_nb)

        self.create_model['fem'] = False
        self.visualizer = visualizer

        # Force pattern
        step = 0.1 if self.visualizer else 0.03
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))
        self.idx_amplitude = 0
        self.force_value = None
        self.idx_zone = 0

    def init_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the mesh model (object will have id = 0)
            self.factory.add_mesh(positions=self.n_visu.position.value.copy(),
                                  cells=self.n_visu.triangles.value.copy(),
                                  at=self.instance_id,
                                  c='orange')
            # Arrows representing the force fields (object will have id = 1)
            self.factory.add_arrows(positions=np.array([0., 0., 0.]),
                                    vectors=np.array([0., 0., 0.]),
                                    c='green',
                                    at=self.instance_id)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Generate a new force
        if self.idx_amplitude == 0:
            self.idx_zone = np.random.randint(0, len(self.cff))
            zone = p_forces.zones[self.idx_zone]
            self.force_value = np.random.uniform(low=-1, high=1, size=(3,)) * p_forces.amplitude[zone]
            self.cff[self.idx_zone].showArrowSize.value = 10 * len(self.cff[self.idx_zone].forces.value)

        # Update current force amplitude
        self.cff[self.idx_zone].force.value = self.force_value * self.amplitudes[self.idx_amplitude]

        # Update force amplitude index
        self.idx_amplitude = (self.idx_amplitude + 1) % len(self.amplitudes)

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Send training data
        self.set_training_data(input=self.compute_input(),
                               ground_truth=np.array([]))

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction in every case.
        return True

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model. Automatically called by DeepPhysX.
        """

        ArmadilloTraining.apply_prediction(self, prediction)
        # Update visualization if required
        if self.visualizer:
            self.update_visual()

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update mesh positions
        self.factory.update_mesh(object_id=0,
                                 positions=self.n_visu.position.value.copy())
        # Update force field
        self.factory.update_arrows(object_id=1,
                                   positions=self.n_surface_mo.position.value[self.cff[self.idx_zone].indices.value],
                                   vectors=0.25 * self.cff[self.idx_zone].forces.value / p_model.scale)
        # Send updated data
        self.update_visualisation()
