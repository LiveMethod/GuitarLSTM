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
    
    # def load_wav_batch(file_paths:List[str], batch_size:int = 4, split:int = 1, preprocess:bool = True) -> Generator[
    #     List[Tuple[int, np.ndarray]], None, None]:
    #     """
    #     Load a batch of wav files
    #     ---
    #     Optionally supports splitting individual files into chunks
    #     Preprocesses audio by default
    #     """
    #     for i in range (0, len(file_paths), batch_size):
    #         batch_files = file_paths[i:i + batch_size]
    #         batch_data = []

    #         for file_path in batch_files:
    #             sr, data = IO.load_wav(file_path)

    #             if preprocess:
    #                 data = data.astype(np.float32).flatten()
    #                 data = Data.normalize(data).reshape(len(data), 1)

    #             if(split > 1):
    #                 chunks = np.split(data[:(len(data)//split)*split], split)
    #                 batch_data.extend(chunks)
    #             else:
    #                 batch_data.append(data)

    #         yield batch_data
    

    """
    Should this be a full feature list for ever sample, or a list of features each all samples long?
    IE if we have gain:4 and tone:6, do we return
    [[6,4], [6,4]....]
    or
    [[6,6...], [4,4...]]
    Partial to the second since it enables easy lookup like features[id]
    
    """
    def load_conditioned_batch(sources:List[Tuple[str, str, str]], batch_size:int = 4, split:int = 1, test_size:float = 0.15) -> Generator[
        List[Tuple[
            List[int],
            Tuple[int, np.ndarray],
            Tuple[int, np.ndarray]
        ]], None, None]:
        """
        Load conditioned batch
        ---
        sources: a list of [conditioning, x_audio, y_audio] tuples
        like [int[], file_path, file_path]

        A conditioned batch is a sequence of audio files paired 
        with their corresopnding conditioning data.
        
        Returns: A generator full of [audio, conditioning] tuples, 
        where audio is a tuple of [sr, data] and conditioning 
        is a 2d array of like list[condition][value]
        """
        for i in range (0, len(sources), batch_size):
            batch_tuples = sources[i:i + batch_size]
            batch_data = []

            for files in batch_tuples:
                c_data, x_path, y_path = files
                sr, x_data = IO.load_wav(x_path)
                sr, y_data = IO.load_wav(y_path)
                x_data = x_data.astype(np.float32).flatten()
                x_data = Data.normalize(x_data).reshape(len(x_data), 1)
                y_data = y_data.astype(np.float32).flatten()
                y_data = Data.normalize(y_data).reshape(len(y_data), 1)

                if(len(x_data) != len(y_data)):
                    raise ValueError(f"Length of x_data and y_data do not match for {x_path} and {y_path}")                

                if(split > 1):
                    x_chunks = np.split(x_data[:(len(x_data)//split)*split], split)
                    y_chunks = np.split(y_data[:(len(y_data)//split)*split], split)
                    if(len(x_chunks) != len(y_chunks)):
                        raise ValueError(f"Number of x_chunks and y_chunks do not match for {x_path} and {y_path}")
                    c_chunks = []
                    for xc in x_chunks:
                        conditioning = []
                        for c in c_data:
                          aaaaaaa = np.repeat(c, len(xc), axis=0)
                          conditioning.append(aaaaaaa)

                    batch_data.extend([c_chunks, x_chunks, y_chunks])
                else:
                    conditioning = []
                    for c in c_data:
                        aaaaaaa = np.repeat(c, len(x_data), axis=0)
                        conditioning.append(aaaaaaa)

                    batch_data.append([c_data, x_data, y_data])
            split_idx = int(len(batch_data) * (1 - test_size))
            train_data = batch_data[:split_idx]
            test_data = batch_data[split_idx:]
            yield train_data, test_data