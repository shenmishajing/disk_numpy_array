from .replacement_strategy import ReplacementStrategy


class SampleNumReplacementStrategy(ReplacementStrategy):
    @staticmethod
    def get_init_state(block):
        return len(block)

    @staticmethod
    def update_state(state):
        return state - 1

    @staticmethod
    def get_block_to_drop(states):
        return min(states, key=lambda x: x[1])[0]
