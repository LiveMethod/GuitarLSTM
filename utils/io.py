from math import exp
from typing import Tuple, Generator, List
from scipy.io import wavfile
import os
from typing import Tuple
import numpy as np
import json

from utils import Data

class IO:
    """
    Misc utils for dealing with the filesystem
    """

    # These live up here so we can grab them from
    # elsewhere without having to pass them around.
    expected_in = 'source_in'
    expected_out = 'source_out'

    def validate_source(source_dir:str) -> Tuple[str, str, str]:
        """
        Validate that requisite folders exist and have usable data.
        Returns the name of the source, and the lists of in and out files.
        """
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"The source directory {source_dir} does not exist.")
        
        source_in_dir = os.path.join(source_dir, IO.expected_in)
        if not os.path.isdir(source_in_dir):
            raise FileNotFoundError(f"Directory {source_dir} is missing folder '{IO.expected_in}'.")

        source_out_dir = os.path.join(source_dir, IO.expected_out)
        if not os.path.isdir(source_out_dir):
            raise FileNotFoundError(f"Directory {source_dir} is missing folder '{IO.expected_out}'.")

        # This will need to widen to accomodate pot conditioninng
        in_files = [os.path.join(source_in_dir, f) for f in os.listdir(source_in_dir) if f.endswith('.wav')]
        out_files = [os.path.join(source_out_dir, f) for f in os.listdir(source_out_dir) if f.endswith('.wav')]

        if len(in_files) == 0:
            raise FileNotFoundError(f"No files found in '{IO.expected_in}' directory.")
        
        if len(out_files) == 0:
            raise FileNotFoundError(f"No files found in '{IO.expected_out}' directory.")
        
        if len(in_files) != len(out_files):
            raise ValueError(f"The number of files in '{IO.expected_in}' and '{IO.expected_out}' directories do not match.")
        
        name = source_dir.split(os.sep)[-1]
        return name, in_files, out_files

    def load_wav(file_path: str) -> Tuple[int, np.ndarray]: # [sr, data]
        """Load single wav file"""
        sr, data = wavfile.read(file_path)
        return sr, data.astype(np.float32)
    
    def load_json(file_path: str) -> dict:
        """Load single json file"""
        with open(file_path) as f:
            return json.load(f)
    
    """
    Should this be a full feature list for ever sample, or a list of features each all samples long?
    IE if we have gain:4 and tone:6, do we return
    [[6,4], [6,4]....]
    or
    [[6,6...], [4,4...]]
    Partial to the second since it enables easy lookup like features[id]
    
    """
    def load_conditioned_batch(
      sources: List[Tuple[List[float], str, str]],
      batch_size: int = 15,
      split: int = 1,
      test_size: float = 0.15
      # return type is a pair (train, test) of data (c, x, y)  .... c is nested, so more like (c[], x, y)
  ) -> Generator[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], None, None]:
      """
      Load conditioned batch.
      """
      for i in range(0, len(sources), batch_size):
          batch_tuples = sources[i:i + batch_size]
          
          # Lists to hold batch data
          conditions_batch = [np.empty((0, 1), dtype=np.float32) for _ in batch_tuples[0][0]]
          x_batch = []
          y_batch = []
          
          for c_data, x_path, y_path in batch_tuples:
              # print(f"Loading ${c_data} and {x_path} and {y_path}")
              sr_x, x_data = IO.load_wav(x_path)
              sr_y, y_data = IO.load_wav(y_path)
              x_data = x_data.astype(np.float32).flatten()
              y_data = y_data.astype(np.float32).flatten()
              x_data = Data.normalize(x_data).reshape(len(x_data), 1)
              y_data = Data.normalize(y_data).reshape(len(y_data), 1)

              if len(x_data) != len(y_data):
                  raise ValueError(f"Length of x_data and y_data do not match for {x_path} and {y_path}")

              # Create conditioning array
              all_c_data = []
              for ci in c_data:
                ci_array = np.full((len(x_data), 1), [ci/100])
                ci_array = ci_array.reshape(len(x_data), 1)
                all_c_data.append(ci_array)

              if split > 1:
                  # # Split data into chunks
                  num_samples = (len(x_data) // split) * split
                  x_splits = np.split(x_data[:num_samples], split)
                  y_splits = np.split(y_data[:num_samples], split)
                  x_batch.extend(x_splits)
                  y_batch.extend(y_splits)
              else:
                  for idx, cb in enumerate(conditions_batch):
                      c_data = all_c_data[idx]
                      conditions_batch[idx] = np.vstack((cb, c_data))
                      
                  # conditions_batch.append(all_c_batches)
                  x_batch.append(x_data)
                  y_batch.append(y_data)
          # Split into training and testing
          split_idx = int(len(x_batch) * (1 - test_size))
          
          
          x_batch = np.concatenate(x_batch, axis=0)
          y_batch = np.concatenate(y_batch, axis=0)
          # Convert lists to NumPy arrays
          c_train = []
          c_test = []
          # print ("x shape, len", x_batch.shape, len(x_batch))
          print(len(conditions_batch))
          # print ("c shape, len", conditions_batch[0].shape, len(conditions_batch[0]))
          for ci in conditions_batch:
              ci_train = ci[:split_idx]
              ci_test = ci[split_idx:]
              # print ("cinp shape, len", ci_train.shape, len(ci_train))
              c_train.append(ci_train)
              c_test.append(ci_test)
          
          # print("train len", len(c_train))
          # print("train[0] len", len(c_train[0]))
          # print(c_train[0])
          # print("x train len", len(x_batch[:split_idx]))
          # print((x_batch[:split_idx]))
          print("###### WILL RETURN TRAINABLE")
          print("x shape", x_batch[:split_idx].shape)
          print("x val", c_train)
          # print("x val", x_batch[:split_idx])

          # Valid like
          # x shape [[-7.0305156e-05]
          #  [-6.6047105e-05]
          #  [-6.2451421e-05]
          #  ...
          #  [ 5.2911364e-03]
          #  [ 5.0643734e-03]
          #  [ 4.7116312e-03]]
          # Processing batch 1...
          train_data = (c_train, x_batch[:split_idx], y_batch[:split_idx])
          test_data = (c_test, x_batch[split_idx:], y_batch[split_idx:])
          yield train_data, test_data