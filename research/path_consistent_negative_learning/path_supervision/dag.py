"""Complete shortest-answer-path membership, shared by graph adapters."""

from typing import Mapping, AbstractSet


def on_shortest_answer_dag(
    head: str,
    tail: str,
    source_distances: Mapping[str, int],
    shortest_path_nodes: AbstractSet[str],
    *,
    transition_cost: int,
) -> bool:
    """Call only for valid transitions of the fixed graph.

    The tail belongs to a complete shortest-answer DAG and the transition
    advances by its full cost. Together these conditions constrain both the
    topic prefix and the answer suffix; local answer-distance decrease alone
    does not establish membership.
    """

    head_distance = source_distances.get(head)
    return (head_distance is not None
            and source_distances.get(tail) == head_distance + transition_cost
            and tail in shortest_path_nodes)
