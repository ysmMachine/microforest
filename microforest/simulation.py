"""DAG manufacturing simulator used to generate bottleneck datasets."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ManufacturingDAG:
    n_tasks: int
    edges: list[tuple[int, int]]
    nominal_cycle_times: list[int]
    initial_buffers: list[int]
    buffer_capacities: list[int]

    @classmethod
    def random(
        cls,
        n_tasks: int,
        edge_probability: float = 0.18,
        seed: int = 7,
        max_cycle_time: int = 8,
        max_initial_buffer: int = 5,
        max_buffer_capacity: int = 12,
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

        nominal_cycle_times = [rng.randint(1, max_cycle_time) for _ in range(n_tasks)]
        buffer_capacities = [rng.randint(max_initial_buffer + 1, max_buffer_capacity) for _ in edges]
        initial_buffers = [rng.randint(0, min(max_initial_buffer, capacity)) for capacity in buffer_capacities]
        return cls(n_tasks, edges, nominal_cycle_times, initial_buffers, buffer_capacities)

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
    def __init__(
        self,
        graph: ManufacturingDAG,
        seed: int = 7,
        cycle_time_jitter: float = 0.15,
        failure_probability: float = 0.01,
        repair_time_range: tuple[int, int] = (2, 6),
        source_release_probability: float = 0.85,
    ):
        self.graph = graph
        self.incoming = graph.incoming
        self.outgoing = graph.outgoing
        self.rng = random.Random(seed)
        self.cycle_time_jitter = cycle_time_jitter
        self.failure_probability = failure_probability
        self.repair_time_range = repair_time_range
        self.source_release_probability = source_release_probability
        self.reset()

    def reset(self) -> None:
        self.time = 0
        self.buffers = list(self.graph.initial_buffers)
        self.remaining = [0 for _ in range(self.graph.n_tasks)]
        self.down_remaining = [0 for _ in range(self.graph.n_tasks)]
        self.production = [0 for _ in range(self.graph.n_tasks)]
        self.starved_count = [0 for _ in range(self.graph.n_tasks)]
        self.blocked_count = [0 for _ in range(self.graph.n_tasks)]
        self.bottlenecks = [0 for _ in range(self.graph.n_tasks)]

    def feature_names(self) -> list[str]:
        names = [f"buffer_{src}_{dst}" for src, dst in self.graph.edges]
        names += [f"buffer_fill_ratio_{src}_{dst}" for src, dst in self.graph.edges]
        names += [f"remaining_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"down_remaining_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"produced_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"starved_count_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"blocked_count_task_{idx}" for idx in range(self.graph.n_tasks)]
        names += [f"nominal_cycle_time_task_{idx}" for idx in range(self.graph.n_tasks)]
        return names

    def snapshot(self) -> list[float]:
        return [
            *[float(value) for value in self.buffers],
            *[
                float(value) / float(capacity)
                for value, capacity in zip(self.buffers, self.graph.buffer_capacities)
            ],
            *[float(value) for value in self.remaining],
            *[float(value) for value in self.down_remaining],
            *[float(value) for value in self.production],
            *[float(value) for value in self.starved_count],
            *[float(value) for value in self.blocked_count],
            *[float(value) for value in self.graph.nominal_cycle_times],
        ]

    def step(self) -> list[int]:
        self.bottlenecks = [0 for _ in range(self.graph.n_tasks)]

        for task_idx, down_time in enumerate(self.down_remaining):
            if down_time > 0:
                self.down_remaining[task_idx] -= 1
                self.bottlenecks[task_idx] = 1

        for task_idx, time_left in enumerate(self.remaining):
            if time_left > 0:
                self.remaining[task_idx] -= 1
                if self.remaining[task_idx] == 0:
                    outgoing_edges = self.outgoing[task_idx]
                    output_space = all(
                        self.buffers[edge_idx] < self.graph.buffer_capacities[edge_idx]
                        for edge_idx in outgoing_edges
                    )
                    if output_space:
                        self.production[task_idx] += 1
                        for edge_idx in outgoing_edges:
                            self.buffers[edge_idx] += 1
                    else:
                        self.remaining[task_idx] = 1
                        self.blocked_count[task_idx] += 1
                        self.bottlenecks[task_idx] = 1

        for task_idx in range(self.graph.n_tasks):
            if self.remaining[task_idx] > 0 or self.down_remaining[task_idx] > 0:
                continue

            if self.rng.random() < self.failure_probability:
                low, high = self.repair_time_range
                self.down_remaining[task_idx] = self.rng.randint(low, high)
                self.bottlenecks[task_idx] = 1
                continue

            incoming_edges = self.incoming[task_idx]
            is_source = not incoming_edges
            has_material = (
                self.rng.random() < self.source_release_probability
                if is_source
                else all(self.buffers[edge_idx] > 0 for edge_idx in incoming_edges)
            )

            outgoing_edges = self.outgoing[task_idx]
            has_output_space = all(
                self.buffers[edge_idx] < self.graph.buffer_capacities[edge_idx]
                for edge_idx in outgoing_edges
            )

            if has_material and has_output_space:
                for edge_idx in incoming_edges:
                    self.buffers[edge_idx] -= 1
                self.remaining[task_idx] = self._sample_cycle_time(task_idx)
            else:
                if not has_material:
                    self.starved_count[task_idx] += 1
                if not has_output_space:
                    self.blocked_count[task_idx] += 1
                self.bottlenecks[task_idx] = 1

        self.time += 1
        return list(self.bottlenecks)

    def _sample_cycle_time(self, task_idx: int) -> int:
        nominal = self.graph.nominal_cycle_times[task_idx]
        if self.cycle_time_jitter <= 0:
            return nominal
        low = max(1, int(round(nominal * (1.0 - self.cycle_time_jitter))))
        high = max(low, int(round(nominal * (1.0 + self.cycle_time_jitter))))
        return self.rng.randint(low, high)


def build_dataset(
    n_tasks: int = 20,
    samples: int = 500,
    horizon: int = 100,
    prediction_window: int = 1,
    warmup: int = 20,
    seed: int = 7,
    edge_probability: float = 0.18,
    cycle_time_jitter: float = 0.15,
    failure_probability: float = 0.01,
    source_release_probability: float = 0.85,
) -> tuple[ManufacturingDAG, list[str], list[str], list[list[float]], list[list[int]]]:
    graph = ManufacturingDAG.random(
        n_tasks=n_tasks,
        edge_probability=edge_probability,
        seed=seed,
    )
    simulator = ManufacturingSimulator(
        graph,
        seed=seed + 1,
        cycle_time_jitter=cycle_time_jitter,
        failure_probability=failure_probability,
        source_release_probability=source_release_probability,
    )

    total_cycles = warmup + samples + horizon + prediction_window + 1
    snapshots: list[list[float]] = []
    labels_by_cycle: list[list[int]] = []
    for _ in range(total_cycles):
        snapshots.append(simulator.snapshot())
        labels_by_cycle.append(simulator.step())

    features: list[list[float]] = []
    labels: list[list[int]] = []
    for idx in range(warmup, warmup + samples):
        features.append(snapshots[idx])
        window = labels_by_cycle[idx + horizon : idx + horizon + prediction_window]
        labels.append([int(any(row[task_idx] for row in window)) for task_idx in range(n_tasks)])

    label_names = [f"y_task_{idx}" for idx in range(n_tasks)]
    return graph, simulator.feature_names(), label_names, features, labels
