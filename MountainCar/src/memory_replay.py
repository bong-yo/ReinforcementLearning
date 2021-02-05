import random


class MemoryReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push_to_memory(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = args
        self.pos = (self.pos + 1) % self.capacity

    def sample_memory(self, size):
        return random.sample(self.memory, size)
