import logging
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, List

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
import math
logger = logging.getLogger(__name__)

def get_partitions_orig(num_samples: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        batch_size: Optional[int] = None,
                        drop_first: int = 0,
                        used_ids: List[int] = None) -> NDArray[np.int64]:
    """
        Adapted from streaming/base/partition/orig.py to manually eliminate sample ids that have 
        been used for training when resuming a run. 
    """
    if num_samples <= drop_first:
        raise ValueError(f'Resuming further into the dataset ({drop_first}) than it has samples ' +
                         f'({num_samples})')

    if num_canonical_nodes < num_physical_nodes:
        if num_physical_nodes % num_canonical_nodes:
            raise ValueError('Either canonical or physical nodes must be evenly divisible by ' +
                             'the other, otherwise striping slices of shards over nodes may ' +
                             'lead to each node downloading all shards')
    elif num_physical_nodes < num_canonical_nodes:
        if num_canonical_nodes % num_physical_nodes:
            raise ValueError('Either canonical or physical nodes must be evenly divisible by ' +
                             'the other, otherwise striping slices of shards over nodes may ' +
                             'lead to each node downloading all shards')

    batch_size = batch_size or 1

    # If drop_first isn't a multiple of num_physical_nodes, round down to make it divisible.
    if drop_first % num_physical_nodes:
        logger.warning(
            '`drop_first` was not divisible by `num_physical_nodes`. Rounding it down ' +
            'to make it divisible.')
        drop_first -= drop_first % num_physical_nodes

    # Divide the full dataset sample range into a sample range per canonical node.
    samples_per_canonical_node = (num_samples + num_canonical_nodes - 1) // num_canonical_nodes
    node_ratio = 0
    padding = 0
    if num_canonical_nodes < num_physical_nodes:
        node_ratio = num_physical_nodes // num_canonical_nodes
        overflow = samples_per_canonical_node % node_ratio
        if overflow:
            padding = node_ratio - overflow
    padded_samples_per_canonical_node = samples_per_canonical_node + padding

    # Create the initial sample ID matrix.
    #
    # ids: (canonical nodes, padded samples per canonical node).
    ids = np.arange(num_canonical_nodes * padded_samples_per_canonical_node, dtype=np.int64)
    ids = ids.reshape(num_canonical_nodes, padded_samples_per_canonical_node)

    # Adjust row offsets to ignore the padding.
    #
    # row_offsets: (canonical nodes, 1).
    row_offsets = np.arange(num_canonical_nodes) * padding
    row_offsets = np.expand_dims(row_offsets, 1)
    ids -= row_offsets

    # Reconfigure where each row starts iterating for irregular-sized rows.
    #
    # row_starts: (canonical nodes, 1).
    row_starts = np.arange(num_canonical_nodes) * num_samples // num_canonical_nodes
    row_starts = np.expand_dims(row_starts, 1)
    ids += row_starts - ids[:, :1]

    # For short rows (length not evenly divisible), repeat the last ID to get even length.
    #
    # row_stops: (canonical nodes, 1).
    row_stops = np.arange(1, 1 + num_canonical_nodes) * num_samples // num_canonical_nodes
    row_stops = np.expand_dims(row_stops, 1)
    are_rows_short = row_stops - row_starts < samples_per_canonical_node
    ids[:, samples_per_canonical_node - 1:samples_per_canonical_node] -= are_rows_short

    # If padding we needed, repeat samples to populate it.
    if padding:
        ids[:, -padding:] = ids[:, -padding - node_ratio + 1 - padding:-padding - node_ratio + 1]

    # Flatten, drop samples that have already been seen, reshape back.
    #
    # ids: (physical nodes, samples per node).
    ids = ids.transpose()
    ids = ids.flatten()
    
    ##### Aded for dynamic batch loading #####
    if used_ids is not None:
        ids = np.setdiff1d(ids, used_ids) # would flatten repeated examples removed repeated examples
        # fill in the shape of ids
        empty_num = num_physical_nodes - len(ids) % num_physical_nodes
        ids = np.concatenate([ids, np.full(empty_num, -1, np.int64)])
    ###########################################

    ids = ids[drop_first:]
    ids = ids.reshape(-1, num_physical_nodes)
    ids = ids.transpose()

    # Interleave the node sample ranges over each node's ranks, padding by repeating the last
    # sample.
    #
    # ids: (physical nodes, samples per rank, ranks per node).
    overflow = ids.shape[1] % ranks_per_node
    if overflow:
        underflow = ranks_per_node - overflow
        last = ids[:, -ranks_per_node - underflow + 1:-ranks_per_node + 1]
        ids = np.concatenate([ids, last], 1)
    ids = ids.reshape(num_physical_nodes, -1, ranks_per_node)

    # Pad with -1 adequately for reshaping across workers.
    #
    # ids: (physical nodes, samples per rank, ranks per node).
    overflow = ids.shape[1] % workers_per_rank
    rounded_num_samples = math.ceil(
        ids.shape[1] / (workers_per_rank * batch_size)) * (workers_per_rank * batch_size)
    overflow = rounded_num_samples - ids.shape[1]
    if overflow:
        last = np.full((num_physical_nodes, overflow, ranks_per_node), -1, np.int64)
        ids = np.concatenate([ids, last], 1)

    # Interleave each rank's padded samples across its workers.
    #
    # ids: (physical nodes, ranks per node, workers per rank, batches per worker, batch size).
    ids = ids.transpose(0, 2, 1)
    ids = ids.reshape(num_physical_nodes, ranks_per_node, -1, workers_per_rank, batch_size)
    return ids.transpose(0, 1, 3, 2, 4)