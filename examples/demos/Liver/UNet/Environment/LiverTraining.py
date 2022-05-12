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

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data and apply prediction.
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
        Compute force vector for the whole surface.
        """

        # Init encoded forces field to zero
        F = np.zeros(self.data_size, dtype=np.double)
        # Encode each force field
        surface_mo = self.f_surface_mo if self.create_model['fem'] else self.n_surface_mo
        for force_field in self.force_field:
            for i in force_field.indices.value:
                # Get the list of nodes composing a cell containing a point from the force field
                p = surface_mo.rest_position.value[i]
                cell = self.regular_grid.cell_index_containing(p)
                # For each node of the cell, encode the force value
                for node in self.regular_grid.node_indices_of(cell):
                    if node < self.nb_nodes_regular_grid and np.linalg.norm(F[node]) == 0.:
                        F[node] = force_field.force.value
        return F

    def compute_output(self):
        """
        Compute displacement vector for the whole surface.
        """

        # Write the position of each point from the sparse grid to the regular grid
        actual_positions_on_regular_grid = np.zeros(self.data_size, dtype=np.double)
        actual_positions_on_regular_grid[self.idx_sparse_to_regular] = self.f_sparse_grid_mo.position.array()
        return np.subtract(actual_positions_on_regular_grid, self.regular_grid_rest_shape)

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model, update visualization data.
        """

        # Reshape to correspond regular grid, transform to sparse grid
        U = np.reshape(prediction, self.data_size)
        U_sparse = U[self.idx_sparse_to_regular]
        self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.array() + U_sparse

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update mesh position
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.f_visu.position.value.copy()})
        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)
