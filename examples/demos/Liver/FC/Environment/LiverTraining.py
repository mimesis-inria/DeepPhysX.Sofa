"""
LiverTraining
Simulation of a Liver with FEM computed simulations.
The SOFA simulation contains two models of a Liver:
    * one to apply forces and compute deformations
    * one to apply the network predictions
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python related imports
import os
import sys

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LiverSofa import LiverSofa, np


class LiverTraining(LiverSofa):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        LiverSofa.__init__(self,
                           root_node=root_node,
                           ip_address=ip_address,
                           port=port,
                           instance_id=instance_id,
                           number_of_instances=number_of_instances,
                           as_tcp_ip_client=as_tcp_ip_client,
                           environment_manager=environment_manager)

        self.create_model['nn'] = True
        self.input_size = None
        self.output_size = None

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called when creating Environment.
        """

        # Add the mesh model (object will have id = 0)
        self.factory.add_object(object_type='Mesh', data_dict={'positions': self.f_visu.position.value.copy(),
                                                               'cells': self.f_visu.triangles.value.copy(),
                                                               'at': self.instance_id,
                                                               'c': 'green'})
        # Return the initial visualization data
        return self.factory.objects_dict

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        # Get the data shape
        self.input_size = self.n_surface_mo.position.value.shape
        self.output_size = self.n_sparse_grid_mo.position.value.shape

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data.
        """

        # Compute training data
        input_array = self.compute_input()
        output_array = self.compute_output()

        # Send training data
        self.set_training_data(input_array=input_array,
                               output_array=output_array)

        # Update visualization
        self.update_visual()

    def compute_input(self):
        """
        Compute force field on the surface.
        """

        # Compute applied force on the surface
        F = np.zeros(self.input_size)
        for cff in self.force_field:
            F[cff.indices.value.copy()] = cff.forces.value.copy()
        return F

    def compute_output(self):
        """
        Compute displacement vector for the whole surface.
        """

        # Compute generated displacement on the grid
        U = self.f_sparse_grid_mo.position.value.copy() - self.f_sparse_grid_mo.rest_position.value.copy()
        return U

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        # Reshape to correspond sparse grid
        U = np.reshape(prediction, self.output_size)
        self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.array() + U

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update mesh positions
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.f_visu.position.value.copy()})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
