"""
LiverPrediction
Simulation of a Liver with NN computed simulations.
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
from LiverTraining import LiverTraining, np
from parameters import p_forces, p_liver


class LiverPrediction(LiverTraining):

    def __init__(self,
                 root_node,
                 ip_address='localhost',
                 port=10000,
                 instance_id=0,
                 number_of_instances=1,
                 as_tcp_ip_client=True,
                 environment_manager=None):

        LiverTraining.__init__(self,
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
        self.nb_forces = p_forces.nb_simultaneous_forces
        self.amplitudes = None
        self.idx_range = 0
        self.forces = []

    def recv_parameters(self, param_dict):
        """
        Exploit received parameters before scene creation.
        """

        # Receive number of forces option
        self.nb_forces = min(param_dict['nb_forces'], self.nb_forces) if 'nb_forces' in param_dict else self.nb_forces
        self.forces = [None] * self.nb_forces

        # Receive visualizer option (either True for Vedo, False for SOFA GUI)
        self.visualizer = param_dict['visualizer'] if 'visualizer' in param_dict else self.visualizer
        step = 0.1 if self.visualizer else 0.05
        self.amplitudes = np.concatenate((np.arange(0, 1, step),
                                          np.arange(1, 0, -step)))

    def send_visualization(self):
        """
        Define and send the initial visualization data dictionary. Automatically called whn creating Environment.
        """

        # Nothing to visualize if the predictions are run in SOFA GUI.
        if self.visualizer:
            # Add the mesh model (object will have id = 0)
            self.factory.add_object(object_type='Mesh', data_dict={'positions': self.n_visu.position.value.copy(),
                                                                   'cells': self.n_visu.triangles.value.copy(),
                                                                   'at': self.instance_id,
                                                                   'c': 'orange'})

            # Arrows representing the force fields (object will have id = 1)
            self.factory.add_object(object_type='Arrows', data_dict={'positions': p_liver.fixed_point,
                                                                     'vectors': [0., 0., 0.],
                                                                     'c': 'green',
                                                                     'at': self.instance_id})

            # Point cloud for sparse grid (object will have id = 2)
            self.factory.add_object(object_type='Points', data_dict={'positions': self.n_sparse_grid_mo.position.value,
                                                                     'r': 2,
                                                                     'c': 'grey',
                                                                     'at': self.instance_id})

        return self.factory.objects_dict

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset force amplitude index
        if self.idx_range == len(self.amplitudes):
            self.idx_range = 0
            self.n_surface_mo.position.value = self.n_surface_mo.rest_position.value

        # Build and set forces vectors
        if self.idx_range == 0:

            # Pick up a random visible surface point, select the points in a centered sphere
            selected_centers = np.empty([0, 3])
            for i in range(self.nb_forces):
                # Pick up a random visible surface point, select the points in a centered sphere
                current_point = self.n_surface_mo.position.value[np.random.randint(len(self.n_surface_mo.position.value))]
                # Check distance to other points
                distance_check = True
                for p in selected_centers:
                    distance = np.linalg.norm(current_point - p)
                    if distance < p_forces.inter_distance_thresh:
                        distance_check = False
                        break
                empty_indices = False

                if distance_check:
                    # Add center to the selection
                    selected_centers = np.concatenate((selected_centers, np.array([current_point])))
                    # Set sphere center
                    self.sphere[i].centers.value = [current_point]
                    # Build force vector
                    if len(self.sphere[i].indices.value) > 0:
                        f = np.random.uniform(low=-1, high=1, size=(3,))
                        self.forces[i] = (f / np.linalg.norm(f)) * p_forces.amplitude
                        self.force_field[i].indices.value = self.sphere[i].indices.array()
                        self.force_field[i].force.value = self.forces[i] * self.amplitudes[self.idx_range]
                    else:
                        empty_indices = True

                if not distance_check or empty_indices:
                    # Reset sphere position
                    self.sphere[i].centers.value = [p_liver.fixed_point.tolist()]
                    # Reset force field
                    self.force_field[i].indices.value = [0]
                    self.force_field[i].force.value = np.array([0.0, 0.0, 0.0])
                    self.forces[i] = np.array([0., 0., 0.])

        # Change force amplitude
        else:
            for i in range(self.nb_forces):
                self.force_field[i].force.value = self.forces[i] * self.amplitudes[self.idx_range]
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

        LiverTraining.apply_prediction(self, prediction)
        # Update visualization if required
        if self.visualizer:
            self.update_visual()

    def update_visual(self):
        """
        Update the visualization data dict.
        """

        # Update mesh positions
        self.factory.update_object_dict(object_id=0, new_data_dict={'position': self.n_visu.position.value.copy()})

        # Update force fields
        position, vector = [], []
        for cff in self.force_field:
            position += list(self.n_surface_mo.position.value[cff.indices.value])
            vector += list(cff.forces.value)
        self.factory.update_object_dict(object_id=1, new_data_dict={'positions': position,
                                                                    'vectors': vector})

        # Update sparse grid positions
        self.factory.update_object_dict(object_id=2,
                                        new_data_dict={'positions': self.n_sparse_grid_mo.position.value})

        # Send updated data
        self.update_visualisation(visu_dict=self.factory.updated_object_dict)


