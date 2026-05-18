"""DAG manufacturing simulator used to generate bottleneck datasets."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ManufacturingDAG:
    n_tasks: int
    edges: list[tuple[int, int]]
    cycle_times: list[int]
    initial_buffers: list[int]

    @classmethod
    def random(
        cls,
        n_tasks: int,
        edge_probability: float = 0.18,
        seed: int = 7,
        max_cycle_time: int = 8,
        max_initial_buffer: int = 5,
    ) -> "ManufacturingDAG":
        rng = random.Random(seed)
        edges: list[tuple[int, int]] = []

        for dst in range(1, n_tasks):
            src = rng.randrange(0, dst)
            edges.append((src, dst))

        for src in range(n_tasks):
            for dst in range(src + 1, n_tasks):
                if (src, dst) not in edges and rng.random() < edge_probability:
                    edges.append((src, dst))

        cycle_times = [rng.randint(1, max_cycle_time) for _ in range(n_tasks)]
        initial_buffers = [rng.randint(0, max_initial_buffer) for _ in edges]
        return cls(n_tasks, edges, cycle_times, initial_buffers)

    @property
    def incoming(self) -> list[list[int]]:
        incoming: list[list[int]] = [[] for _ in range(self.n_tasks)]
        for edge_idx, (_src, dst) in enumerate(self.edges):
            incoming[dst].append(edge_idx)
        return incoming

    @property
    def outgoing(self) -> list[list[int]]:
        outgoing: list[list[int]] = [[] for _ in range(self.n_tasks)]
        for edge_idx, (src, _dst) in enumerate(self.edges):
            outgoing[src].append(edge_idx)
        return outgoing


class ManufacturingSimulator:
    def __init__(self, graph: ManufacturingDAG):
        self.graph = graph
        self.incoming = graph.incoming
        self.outgoing = graph.outgoing
        self.reset()

    def reset(self) -> None:
        self.time = 0
        self.buffers = list(self.graph.initial_buffers)
        self.remaining = [0 for _ in range(self.graph.n_tasks)]
        self.production = [0 for _ in range(self.graph.n_tasks)]
        self.bottlenecks = [0 for _ in range(self.graph.n_tasks)]

    def feature_names(self) -> list[str]:
        names = [f"buffer_{src}_{dst}" for src, dst in self.graph.edges]
        names += [f"remaining_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"produced_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"cycle_time_task_{idx}" for idx in range(self.graph.n_tasks)]
        return names

    def snapshot(self) -> list[float]:
        return [
            *[float(value) for value in self.buffers],
            *[float(value) for value in self.remaining],
            *[float(value) for value in self.production],
            *[float(value) for value in self.graph.cycle_times],
        ]

    def step(self) -> list[int]:
        self.bottlenecks = [0 for _ in range(self.graph.n_tasks)]

        for task_idx, time_left in enumerate(self.remaining):
            if time_left > 0:
                self.remaining[task_idx] -= 1
                if self.remaining[task_idx] == 0:
                    self.production[task_idx] += 1
                    for edge_idx in self.outgoing[task_idx]:
                        self.buffers[edge_idx] += 1

        for task_idx in range(self.graph.n_tasks):
            if self.remaining[task_idx] > 0:
                continue

            incoming_edges = self.incoming[task_idx]
            is_source = not incoming_edges
            has_material = is_source or all(self.buffers[edge_idx] > 0 for edge_idx in incoming_edges)

            if has_material:
                for edge_idx in incoming_edges:
                    self.buffers[edge_idx] -= 1
                self.remaining[task_idx] = self.graph.cycle_times[task_idx]
            else:
                self.bottlenecks[task_idx] = 1

        self.time += 1
        return list(self.bottlenecks)


def build_dataset(
    n_tasks: int = 20,
    samples: int = 500,
    horizon: int = 100,
    warmup: int = 20,
    seed: int = 7,
) -> tuple[ManufacturingDAG, list[str], list[str], list[list[float]], list[list[int]]]:
    graph = ManufacturingDAG.random(n_tasks=n_tasks, seed=seed)
    simulator = ManufacturingSimulator(graph)

    total_cycles = warmup + samples + horizon + 1
    snapshots: list[list[float]] = []
    labels_by_cycle: list[list[int]] = []
    for _ in range(total_cycles):
        snapshots.append(simulator.snapshot())
        labels_by_cycle.append(simulator.step())

    features: list[list[float]] = []
    labels: list[list[int]] = []
    for idx in range(warmup, warmup + samples):
        features.append(snapshots[idx])
        labels.append(labels_by_cycle[idx + horizon])

    label_names = [f"y_task_{idx}" for idx in range(n_tasks)]
    return graph, simulator.feature_names(), label_names, features, labels

