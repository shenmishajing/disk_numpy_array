from abc import ABC


class ReplacementStrategy(ABC):
    @staticmethod
    def get_init_state(block):
        raise NotImplementedError

    @staticmethod
    def update_state(state):
        raise NotImplementedError

    @staticmethod
    def get_block_to_drop(states):
        raise NotImplementedError
