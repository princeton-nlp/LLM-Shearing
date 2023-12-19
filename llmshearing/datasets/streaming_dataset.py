# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A mid-epoch-resumable streaming/caching pytorch IterableDataset."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from filelock import FileLock
from numpy.typing import NDArray
from streaming.base.constant import EPOCH_DATA, EPOCH_SHAPE, RESUME
from streaming.base.dataset import StreamingDataset
from streaming.base.shared import SharedMemory, _get_path
from streaming.base.shuffle import get_shuffle
from streaming.base.stream import Stream
from streaming.base.world import World

from llmshearing.datasets.partition import get_partitions_orig

# An arbitrary time in the future, used for cold shard eviction.
NEVER = np.iinfo(np.uint64).max
logger = logging.getLogger(__name__)


def generate_work(dataset: StreamingDataset, 
                  world: World, 
                  epoch: int, 
                  used_domain_ids: List[List[int]]) -> NDArray[np.int64]:
    """Generate this epoch's arrangement of samples for each stream (domain) of data.

    Args:
        dataset (StreamingDataset): Dataset to generate the partition for.
        world (World): World state.
        epoch (int): Which epoch it is.
        sample_in_epoch (int): Where we are in the epoch.

    Returns:
        List[List[int]]: The epoch for each domain of data (num physical nodes, 
        ranks per node, workers per rank, batches per worker, batch size).
    """
    assert epoch == 0, "Currently only supports dynamic loading from each domain for once."
    # Ensure that num_canonical_nodes has been set.
    if dataset.num_canonical_nodes is None:
        raise RuntimeError(f'`num_canonical_nodes` can never be None. ' +
                           f'Provide a positive integer.')

    # First, for each stream, sample each shard of the stream according to
    # stream proportions/repeats/samples. We obtain the resampled size of each shard
    # in the stream and a mapping from the training "big" sample ID to the underlying
    # shard "small" sample ID. Then, we also partition each stream's samples over
    # nodes/devices/workers. We handle sample_in_epoch (for resumption) at the end.
    partition_per_stream = []

    batch_size = dataset.batch_size or 1

    for stream_id, stream in enumerate(dataset.streams):
        # choose always == len(stream)
        shuffle_units, small_per_big = dataset.resample_streams(epoch, stream_id)
        samples_in_stream = len(small_per_big)
        
        used_stream_ids = np.array(used_domain_ids[stream_id])
        used_stream_ids -= dataset.sample_offset_per_stream[stream_id]
        shuffle_block_portion = int(dataset.shuffle_block_size * stream.proportion)
        if dataset.shuffle:
            mapping = get_shuffle(dataset.shuffle_algo, shuffle_units,
                                    dataset.num_canonical_nodes, dataset.shuffle_seed, epoch, shuffle_block_portion)
            reverse_mapping = {mapping[k]: k for k in range(len(mapping))}
            used_stream_ids = np.array([reverse_mapping[used_stream_ids[j]] for j in range(len(used_stream_ids))], 
                                dtype=np.int64)
            del reverse_mapping
                     
        # check 
        stream_partition = get_partitions_orig(samples_in_stream,
                                               dataset.num_canonical_nodes, world.num_nodes,
                                               world.ranks_per_node, world.workers_per_rank, batch_size,
                                               0, used_stream_ids)
        if dataset.shuffle:
            # Ratio of stream's shuffle block size to overall shuffle block size should be the
            # same as the ratio of the stream's samples to overall samples.
            # This ensures that the overall training shuffle block size is still approximately
            # equal to what is set by the user, and allows for reasoning about cache_limit as well.
            stream_shuffle = get_shuffle(dataset.shuffle_algo, shuffle_units,
                                         dataset.num_canonical_nodes, dataset.shuffle_seed, epoch,
                                         shuffle_block_portion)
            stream_partition = np.where(stream_partition != -1, stream_shuffle[stream_partition],
                                        -1)
        # The small_per_big array already corresponds to indices of samples per shard of each
        # stream. So each sample ID in the stream's partition already corresponds to the sample ID
        # in the right shard.
        partition_per_stream.append(
            np.where(stream_partition != -1, small_per_big[stream_partition], -1))
        assert np.intersect1d(partition_per_stream[-1].flatten(), np.array(used_domain_ids[stream_id])).size == 0
    return partition_per_stream

class DynamicStreamingDataset(StreamingDataset):
    """ This is an inherited class from StreamingDataset to support dynamic loading from different data streams (domains). """
    def __init__(self,
                 *,
                 local: Optional[str] = None,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 set_names: List[str] = None,
                 proportion: List[float] = None) -> None:
         
        streams = [Stream(local=local, split=set_name, repeat=1.0) for set_name in set_names]
        super().__init__(streams=streams, 
                         split=None,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed) 

        self.set_names = set_names
        self.used_num_samples_per_stream = [0 for _ in range(self.num_streams)]
        self.proportion = list(proportion)
    
    def update_proportion(self, proportion: List[float]) -> None:
        self.proportion = proportion
         
    def state_dict(self, used_sample_ids: List[List[int]], from_beginning: bool) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Our stock StreamingDataLoader counts samples from start of training (from_beginning=false).
        However, if you are always counting from the start of the epoch, set from_beginning=true.

        Args:
            used_sample_ids: Used sample ids for each stream.
            from_beginning: Whether we are counting samples from the start of this epoch, or
                the start of just this potentially resumed training run this epoch.

        Returns:
            Dict[str, Any]: The state.
        """
        world = World()
        epoch = self.next_epoch - 1
        epoch, offset = self._resume(world, epoch)

        assert from_beginning
        domain_sample_in_epoch = [len(used_sample_ids[i]) for i in range(len(self.set_names))] 
        
        return {
            'epoch': epoch,
            'domain_sample_in_epoch': domain_sample_in_epoch,
            'used_sample_ids': used_sample_ids,
            'num_canonical_nodes': self.num_canonical_nodes,
            'proportion': self.proportion,
            'shuffle_seed': self.shuffle_seed
        }

    def _resume(self, world: World, epoch: int) -> Tuple[int, int]:
        """Either resume from checkpoint or start at the beginning.

        Args:
            world (World): World state.
            epoch (int): What epoch we think it is (pre-checkpoint).

        Returns:
            Tuple[int, int]: What epoch this is, and sample offset in that epoch.
        """
        # Get the resume state, if it exists.
        name = _get_path(self._shm_prefix_int, RESUME)
        try:
            shm = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            # There is nothing to resume.
            if not self.num_canonical_nodes:
                self.num_canonical_nodes = world.num_nodes * 64
            self._set_predownload()
            return epoch, np.array([[] for _ in range(self.num_streams)], dtype=np.int64)

        # SharedMemory buffers may contain additional null bytes at the end.
        buf = bytes(shm.buf)
        index = buf.find(b'\0')
        buf = buf[:index] if index != -1 else buf
        obj = json.loads(buf.decode('utf-8'))

        # Check if the resume state is stale.
        if obj['epoch'] < epoch:
            if not self.num_canonical_nodes:
                self.num_canonical_nodes = world.num_nodes * 64
            self._set_predownload()
            return epoch, 0

        # Load the correct resumption meta data.
        epoch = obj['epoch']
        assert epoch == 0, "Currently only supports dynamic loading from each domain for once."
        used_sample_ids = obj['used_sample_ids']
        self.num_canonical_nodes = obj['num_canonical_nodes']
        self.shuffle_seed = obj['shuffle_seed']
        self._set_predownload()

        return epoch, used_sample_ids
    
    def _resume_incr_epoch(self, world: World) -> Tuple[int, int]:
        """Start or resume training, pre-incrementing the next epoch.

        Args:
            world (World): World state.

        Returns:
            Tuple[int, List[List[int]]]: What epoch this is, and used sample ids per stream
        """
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Either resume from checkpoint, or start from scratch.
        presumed_epoch = self.next_epoch
        epoch, used_sample_ids = self._resume(world, presumed_epoch)

        # Wait for everyone to get the epoch above.
        self._shared_barrier(world.workers_per_node)

        # Set the new next epoch.
        if world.is_local_leader:
            self.next_epoch = epoch + 1

        return epoch, used_sample_ids
     
    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        We just save the state to shared memory for workers to pick up when __iter__ is next
        called. We use shm because changes to this copy of the dataset wouldn't be picked up by
        persistent workers.

        Args:
            obj (Dict[str, Any]): The state.
        """
        self.update_proportion(obj["proportion"])
        print("Loaded proportion", obj["proportion"])
        assert "used_sample_ids" in obj # used_sample_ids is not an entry in this class
        
        from copy import deepcopy
        obj_copy = deepcopy(obj) 
        
        name = _get_path(self._shm_prefix_int, RESUME)
        data = json.dumps(obj_copy, sort_keys=True).encode('utf-8')
        # Some platforms choose to allocate chunks of memory based upon that platform's memory page
        # size, hence the exact size of the shared memory block that was returned may be larger
        # than what was requested.
        self._resume_shm = SharedMemory(name=name, size=len(data))
        self._resume_shm.buf[:len(data)] = data
        
    def _share_work(self, sample_ids_per_stream: List[List[int]]) -> List[Tuple[SharedMemory, SharedMemory]]:
        """Put an epoch's sample ordering into shared memory for each stream (domain).

        Args:
            sample_ids_per_stream (NDArray[np.int64]): Sample IDs.

        Returns:
            Tuple[SharedMemory, SharedMemory]: Shared memory arrays containing shape and data.
        """
        ndim = 5
        shape_shms = []; data_shms = []
        for stream_id, sample_ids in enumerate(sample_ids_per_stream):
            # Validate shape.
            if sample_ids.ndim != ndim:
                raise ValueError(f'Sample IDs must be of {ndim}D shape (num physical nodes, ' +
                                f'ranks per node, workers per rank, batches per worker, ' +
                                f'batch size). Instead, found as {sample_ids.ndim}D shape.')

            # Save the generated epoch shape to shared memory.
            name = _get_path(self._shm_prefix_int, EPOCH_SHAPE + f"_{stream_id}")
            size = ndim * np.int64().nbytes
            shape_shm = SharedMemory(name=name, create=True, size=size, auto_cleanup=False)
            shape_shm.buf[:size] = np.array(sample_ids.shape, np.int64).tobytes()

            # Save the generated epoch data to shared memory.
            name = _get_path(self._shm_prefix_int, EPOCH_DATA + f"_{stream_id}")
            size = sample_ids.size * np.int64().nbytes
            data_shm = SharedMemory(name=name, create=True, size=size, auto_cleanup=False)
            data_shm.buf[:size] = sample_ids.tobytes()
            
            shape_shms.append(shape_shm)
            data_shms.append(data_shm)

        return shape_shms, data_shms
    
    def _attach_work(self) -> Tuple[NDArray[np.int64], SharedMemory, SharedMemory]:
        """Get an epoch's sample ordering from shared memory.

        Returns:
            NDArray[np.int64]: Sample IDs.
        """
        ndim = 5

        sample_ids_per_stream, shape_shms, data_shms = [], [], []
        for stream_id in range(self.num_streams):
            # Load the generated epoch shape from shared memory.
            name = _get_path(self._shm_prefix_int, EPOCH_SHAPE + f"_{stream_id}")
            size = ndim * np.int64().nbytes
            shape_shm = SharedMemory(name=name, create=False, size=size, auto_cleanup=False)
            shape = tuple(np.ndarray(5, buffer=shape_shm.buf, dtype=np.int64))

            # Attach to the generated epoch data in shared memory.
            name = _get_path(self._shm_prefix_int, EPOCH_DATA + f"_{stream_id}")
            size = int(np.prod(shape)) * np.int64().nbytes
            data_shm = SharedMemory(name=name, create=False, size=size, auto_cleanup=False)
            sample_ids = np.ndarray(shape, buffer=data_shm.buf, dtype=np.int64)
            sample_ids_per_stream.append(sample_ids)
            shape_shms.append(shape_shm)
            data_shms.append(data_shm)
        return sample_ids_per_stream, shape_shms, data_shms

    def _get_work(self, world: World, epoch: int, used_domain_ids: List[List[int]]) -> NDArray[np.int64]:
        """Get this worker's partition of this epoch's sample space for each stream (domain).

        Args:
            world (World): World state.
            epoch (int): Which epoch it is.
            sample_in_epoch (List(NDArray[np.int64])): Where we are in the epoch.

        Returns:
            Optional[NDArray[np.int64]]: Our partition of the epoch.
        """
        # Lazily create the shared barrier's FileLock, which contains a threading Lock, which is
        # unpickleable.
        if not hasattr(self._shared_barrier, 'lock'):
            self._shared_barrier.lock = FileLock(self._shared_barrier.filelock_path)

        # Do expensive work that may use a lot of cores/memory just once, in the local leader.
        if world.is_local_leader:
            sample_ids_per_stream = generate_work(self, world, epoch, used_domain_ids)
            shape_shms, data_shms = self._share_work(sample_ids_per_stream)
            self._shared_barrier(world.workers_per_node)
        else:
            self._shared_barrier(world.workers_per_node)
            sample_ids_per_stream, shape_shms, data_shms = self._attach_work()

        # Each worker gets their portion of the work.
        sample_ids_per_stream = [sample_ids_per_stream[stream_id]
                                 [world.node, world.rank_of_node, world.worker_of_rank].flatten() 
                                 for stream_id in range(self.num_streams)]

        self._shared_barrier(world.workers_per_node)

        # Now clean up after ourselves.
        for shape_shm, data_shm in zip(shape_shms, data_shms):
            shape_shm.cleanup()
            data_shm.cleanup()

        return sample_ids_per_stream
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Exit the threads that are pre-downloading and iterating the shards for previous epoch, if
        # it exists.
        if hasattr(self, '_iterator'):
            self._iterator.exit()

        # For exception handling.
        if not hasattr(self, '_executor'):
            self._executor = ThreadPoolExecutor()
        if not hasattr(self, '_event'):
            self._event = Event()
        elif self._event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, used_sample_ids = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids_per_stream = self._get_work(world, epoch, used_sample_ids)
       
        # Currently only supports dynamically loading data from each domain for once. 
        # Issues could occur if one domain of data is used up. 
        while True:
            proportion = self.proportion
            stream_id = np.random.choice(range(self.num_streams), 1, p=proportion)[0].item()
            domain_sample_id = sample_ids_per_stream[stream_id]
            domain_sample_id = domain_sample_id[self.used_num_samples_per_stream[stream_id] % self.samples_per_stream[stream_id]]
            self.used_num_samples_per_stream[stream_id] += 1
            yield self[domain_sample_id]

class TextDynamicStreamingDataset(DynamicStreamingDataset):
    """ 
        A dataset to load data dynamically from different domains
        Adapted from https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/text_data.py#L21
    """

    def __init__(self,
                 local: str,
                 max_seq_len: int,
                 shuffle: bool = False,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = 128,
                 batch_size: Optional[int] = None,
                 set_names: List[str] = None,
                 proportion: List = None,
                 is_uint16: bool = False):

        # Build Dataset
        super().__init__(local=local,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         set_names=set_names,
                         proportion=proportion)
        
        # Token ids are in a uint16 format to save memory
        self.is_uint16 = is_uint16
        self.max_seq_len = max_seq_len

    def _read_binary_tokenized_sample(self, sample):
        if self.is_uint16:
            a = np.frombuffer(sample['tokens'], dtype="B").view(
                dtype=np.uint16).astype(np.int64)
            tokens = torch.from_numpy(a[:self.max_seq_len].copy())
        else:
            tokens = torch.from_numpy(np.frombuffer(sample['tokens'], dtype=np.int64)[:self.max_seq_len].copy())
        return tokens

    def get_sample(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return sample

    # updated
    def __getitem__(self, idx: Union[int, Tuple]) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        token_sample = self._read_binary_tokenized_sample(sample)
        return {"input_ids": token_sample, "set": sample["set"], "idx": idx} 

class TextStreamingDataset(StreamingDataset):
    """ 
        A dataset to load fixed data, a simplied version of 
        Adapted from https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/text_data.py#L21
    """
    def __init__(self,
                 local: str,
                 split: str,
                 max_seq_len: int,
                 shuffle: bool = False,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = 128,
                 batch_size: Optional[int] = None,
                 is_uint16: bool = False):

        # Build Dataset
        super().__init__(local=local,
                         split=split,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size)
        
        # Token ids are in a uint16 format to save memory
        self.is_uint16 = is_uint16
        self.max_seq_len = max_seq_len

    def _read_binary_tokenized_sample(self, sample):
        if self.is_uint16:
            a = np.frombuffer(sample['tokens'], dtype="B").view(
                dtype=np.uint16).astype(np.int64)
            tokens = torch.from_numpy(a[:self.max_seq_len].copy())
        else:
            tokens = torch.from_numpy(np.frombuffer(sample['tokens'], dtype=np.int64)[:self.max_seq_len].copy())
        return tokens

    def get_sample(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return sample

    # updated
    def __getitem__(self, idx: Union[int, Tuple]) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        token_sample = self._read_binary_tokenized_sample(sample)
        return {"input_ids": token_sample, "set": sample["set"], "idx": idx} 