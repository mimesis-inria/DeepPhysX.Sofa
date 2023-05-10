from typing import Type, Tuple, Dict, Any, Union, Optional

from SSD.SOFA.Rendering.UserAPI import UserAPI, Database
from DeepPhysX.Sofa.Environment.SofaEnvironment import SofaEnvironment
from DeepPhysX.Core.Environment.BaseEnvironmentController import BaseEnvironmentController


class SofaEnvironmentController(BaseEnvironmentController):

    def __init__(self,
                 environment_class: Type[SofaEnvironment],
                 environment_kwargs: Dict[str, Any],
                 environment_ids: Tuple[int, int] = (1, 1),
                 use_database: bool = True):

        BaseEnvironmentController.__init__(self,
                                           environment_class=environment_class,
                                           environment_kwargs=environment_kwargs,
                                           environment_ids=environment_ids,
                                           use_database=use_database)
        self.__environment: Optional[SofaEnvironment] = None
        self.__environment_class: Type[SofaEnvironment] = environment_class
        self.__environment_kwargs: Dict[str, Any] = environment_kwargs
        self.__visualization_factory: Optional[UserAPI] = None

    @property
    def environment(self) -> SofaEnvironment:
        return self.__environment

    @property
    def visualization_factory(self) -> Optional[UserAPI]:
        return self.__visualization_factory

    def create_environment(self) -> SofaEnvironment:

        self.__environment: SofaEnvironment = self.__environment_class(**self.__environment_kwargs,
                                                                       environment_controller=self)
        self.__environment.create()
        self.__environment.init()
        self.__environment.init_database()
        return self.__environment

    def create_visualization(self,
                             visualization_db: Union[Database, Tuple[str, str]],
                             produce_data: bool = True) -> None:

        if type(visualization_db) == list:
            self.__visualization_factory = UserAPI(root=self.__environment.root,
                                                   database_dir=visualization_db[0],
                                                   database_name=visualization_db[1],
                                                   idx_instance=self.environment_ids[0] - 1,
                                                   non_storing=not produce_data)
        else:
            self.__visualization_factory = UserAPI(root=self.__environment.root,
                                                   database=visualization_db,
                                                   idx_instance=self.environment_ids[0] - 1,
                                                   non_storing=not produce_data)
        self.__environment.init_visualization()
