from typing import Optional
import Sofa

from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig


class SofaPrediction(Sofa.Core.Controller, BasePrediction):

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: SofaEnvironmentConfig,
                 database_config: Optional[BaseDatabaseConfig] = None,
                 session_dir: str = 'session',
                 session_name: str = 'training',
                 step_nb: int = -1,
                 record: bool = False,
                 *args, **kwargs):
        """
        SofaPrediction is a pipeline defining the running process of an artificial neural network.
        It provides a highly tunable learning process that can be used with any machine learning library.

        :param network_config: Configuration object with the parameters of the Network.
        :param environment_config: Configuration object with the parameters of the Environment.
        :param database_config: Configuration object with the parameters of the Database.
        :param session_dir: Relative path to the directory which contains sessions repositories.
        :param session_name: Name of the new the session repository.
        :param step_nb: Number of simulation step to play.
        :param record: If True, prediction data will be saved in a dedicated Database.
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        BasePrediction.__init__(self,
                                network_config=network_config,
                                database_config=database_config,
                                environment_config=environment_config,
                                session_name=session_name,
                                session_dir=session_dir,
                                step_nb=step_nb,
                                record=record)

        self.load_samples = environment_config.load_samples
        self.prediction_begin()

        # Add the Pipeline before the Environment in the root node (Events are executed sequentially)
        env = self.data_manager.environment_manager.environment
        self.root = env.root

        if self.load_samples:
            self.root.removeObject(env)
            self.root.addObject(self)
            self.root.addObject(env)
        else:
            self.root.addObject(self)

    def onAnimateBeginEvent(self, _):
        if self.load_samples:
            sample_id = self.data_manager.load_sample()
            self.data_manager.environment_manager.environment._get_training_data(sample_id)

    def onAnimateEndEvent(self, _):
        """
        Called within the Sofa pipeline at the end of the time step.
        """

        if self.prediction_condition():
            self.sample_begin()
            self.predict()
            self.sample_end()

    def predict(self) -> None:
        """
        Pull the data from the manager and return the prediction.
        """

        self.data_manager.get_data(epoch=0,
                                   animate=False,
                                   load_samples=not self.load_samples)

    def close(self) -> None:
        """
        Manually trigger the end of the Pipeline.
        """

        self.prediction_end()
