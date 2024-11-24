from math import exp
import os
from typing import Tuple

class IO:
    """
    Misc utils for dealing with the filesystem
    """

    # These live up here so we can grab them from
    # elsewhere without having to pass them around.
    expected_in = 'source_in'
    expected_out = 'source_out'

    def validate_source(source_dir: str) -> Tuple[str, str, str]:
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
    
