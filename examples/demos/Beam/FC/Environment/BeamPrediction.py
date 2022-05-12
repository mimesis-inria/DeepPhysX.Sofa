"""
BeamPrediction
Simulation of a Beam with NN computed simulations.
The SOFA simulation contains the model used to apply the network predictions.
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python imports
import os
import sys

# Working session imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeamTraining import BeamTraining, p_grid, np


class BeamPrediction(BeamTraining):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        BeamTraining.__init__(self,
                              root_node=root_node,
                              ip_address=ip_address,
                              port=port,
                              instance_id=instance_id,
                              number_of_instances=number_of_instances,
                              as_tcp_ip_client=as_tcp_ip_client,
                              environment_manager=environment_manager)

        self.create_model['fem'] = False
        self.visualizer = False

        # Force pattern
        self.amplitudes = None
        self.idx_range = 0
        self.force_value = None
        self.indices_value = None

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        # Receive visualizer option (either True for Vedo, False for SOFA GUI)
        self.visualizer = param_dict['visualizer'] if 'visualizer' in param_dict else self.visualizer
        step = 0.1 if self.visualizer else 0.03
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the mesh model (object will have id = 0)
            self.factory.add_object(object_type='Mesh', data_dict={'positions': self.n_visu.position.value.copy(),
                                                                   'cells': self.n_visu.triangles.value.copy(),
                                                                   'at': self.instance_id,
                                                                   'c': 'orange'})

            # Arrows representing the force fields (object will have id = 1)
            self.factory.add_object(object_type='Arrows', data_dict={'positions': [0, 0, 0],
                                                                     'vectors': [0., 0., 0.],
                                                                     'c': 'green',
                                                                     'at': self.instance_id})

        return self.factory.objects_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset force amplitude index
        if self.idx_range == len(self.amplitudes):
            self.idx_range = 0

        # Create a new non-empty random box ROI, select nodes of the surface
        if self.idx_range == 0:

            # Define random box
            side = np.random.randint(0, 6)
            x_min = p_grid.min[0] if side == 0 else np.random.randint(p_grid.min[0], p_grid.max[0] - 10)
            x_max = p_grid.max[0] if side == 1 else np.random.randint(x_min + 10, p_grid.max[0] + 1)
            y_min = p_grid.min[1] if side == 2 else np.random.randint(p_grid.min[1], p_grid.max[1] - 10)
            y_max = p_grid.max[1] if side == 3 else np.random.randint(y_min + 10, p_grid.max[1] + 1)
            z_min = p_grid.min[2] if side == 4 else np.random.randint(p_grid.min[2], p_grid.max[2] - 10)
            z_max = p_grid.max[2] if side == 5 else np.random.randint(z_min + 10, p_grid.max[2] + 1)

            # Set the new bounding box
            self.root.nn.removeObject(self.cff_box)
            self.cff_box = self.root.nn.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
                                                  box=[x_min, y_min, z_min, x_max, y_max, z_max])
            self.cff_box.init()

            # Get the intersection with the surface
            indices = list(self.cff_box.indices.value)
            indices = list(set(indices).intersection(set(self.idx_surface)))

            # Create a random force vector
            F = np.random.uniform(low=-1, high=1, size=(3,))
            K = np.random.randint(20, 30)
            F = K * (F / np.linalg.norm(F))
            # Keep value
            self.force_value = F
            self.indices_value = indices

        # Update force field
        F = self.amplitudes[self.idx_range] * self.force_value
        self.root.nn.removeObject(self.cff)
        self.cff = self.root.nn.addObject('ConstantForceField', name='CFF', showArrowSize=0.5,
                                          indices=self.indices_value, force=list(F))
        self.cff.init()
        self.idx_range += 1

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
        """

        # Compute training data
        input_array = self.compute_input()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=np.array([]))

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the network prediction even if the solver diverged.
        return True

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        BeamTraining.apply_prediction(self, prediction)
        # Update visualization if required
        if self.visualizer:
            self.update_visual()

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update mesh positions
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.n_visu.position.value.copy()})

        # Update force field
        position = list(self.n_visu.position.value[self.cff.indices.value])
        vector = list(0.25 * self.cff.forces.value)
        self.factory.update_object_dict(object_id=1, new_data_dict={'positions': position,
                                                                    'vectors': vector})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
