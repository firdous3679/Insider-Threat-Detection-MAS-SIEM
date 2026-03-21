"""Minimal local Mesa compatibility layer for this repository's simulation."""

from __future__ import annotations

import random


class Model:
    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.agents = []
        self._next_id = 1

    def _register_agent(self, agent):
        self.agents.append(agent)

    def next_id(self):
        uid = self._next_id
        self._next_id += 1
        return uid


class Agent:
    def __init__(self, model):
        self.model = model
        self.random = model.random
        self.unique_id = model.next_id()
        model._register_agent(self)
