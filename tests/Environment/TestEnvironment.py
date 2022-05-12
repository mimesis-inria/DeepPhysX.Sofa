from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


class TestEnvironment(SofaEnvironment):

    def __init__(self, root_node, ip_address='localhost', port=10000, instance_id=1, number_of_instances=1,
                 as_tcp_ip_client=True, environment_manager=None):
        SofaEnvironment.__init__(self, root_node=root_node, ip_address=ip_address, port=port, instance_id=instance_id,
                                 number_of_instances=number_of_instances, as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

        # Check init functions call
        self.call_create = False
        self.call_init = False
        self.call_step = False
        # Parameters to receive
        self.parameters = {}

    def create(self):
        # Assert method is called
        self.call_create = True

    def onSimulationInitDoneEvent(self, event):
        # Assert method is called
        self.call_init = True

    def recv_parameters(self, param_dict):
        self.parameters = param_dict

    def send_parameters(self):
        dict_to_return = {}
        for key, value in self.parameters.items():
            dict_to_return[key] = value * int(key[-1])
        return dict_to_return

    def onAnimateBeginEvent(self, event):
        # Assert method is called
        self.call_step = True
