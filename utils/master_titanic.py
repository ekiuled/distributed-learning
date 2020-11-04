from utils.consensus_titanic import ConsensusNode
from numpy import array_split


class MasterNode:
    def __init__(self, node_names,
                 weights: dict,
                 X_train, y_train, X_test, y_test,
                 stat_step=50,
                 epoch=200, epoch_len=391, epoch_cons_num=0):
        self.node_names = node_names
        self.weights: dict = weights
        self.network = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.stat_step = stat_step
        self.epoch: int = epoch
        self.epoch_len: int = epoch_len
        self.epoch_cons_num: int = epoch_cons_num

    def initialize_nodes(self):
        X_train_parts = array_split(self.X_train, len(self.node_names))
        y_train_parts = array_split(self.y_train, len(self.node_names))

        self.network = {name: ConsensusNode(name=name,
                                            X_train=X,
                                            y_train=y,
                                            X_test=self.X_test,
                                            y_test=self.y_test,
                                            stat_step=self.stat_step,
                                            weights=self.weights[name])
                        for name, X, y in zip(self.node_names, X_train_parts, y_train_parts)}

        for node_name, node in self.network.items():
            node.set_neighbors({neighbor_name: self.network[neighbor_name]
                                for neighbor_name in self.weights[node_name]
                                if neighbor_name != node_name})

    def start_consensus(self):
        for ep in range(1, self.epoch + 1):
            for it in range(self.epoch_len):
                for node_name, node in self.network.items():
                    node.fit_step(ep)
                if ep >= self.epoch_cons_num:
                    for node_name, node in self.network.items():
                        node.ask_params()

                    for node_name, node in self.network.items():
                        node.update_params()
