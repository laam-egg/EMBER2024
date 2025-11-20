import sys
from pathlib import Path
import time
import os
import hashlib
import click
import csv
import functools
from contextlib import contextmanager

# --- Lazy Loaders ---
@functools.lru_cache(maxsize=1)
def get_model():
    import lightgbm as lgb
    # Ensure 'resource' is available here or pass the path in
    return lgb.Booster(model_file=str(resource("models/EMBER2024_all.model")))

@functools.lru_cache(maxsize=1)
def get_extractor():
    import thrember
    return thrember.PEFeatureExtractor()

def resource(path):
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS) # type: ignore
    else:
        base = Path('.')
    return base / path

@contextmanager
def get_output_stream(output_path):
    if not output_path:
        yield sys.stdout
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            yield f

# --- Core Logic (Extracted from CLI) ---

def process_single_file(filename: str, max_size_bytes: int):
    """
    Analyzes a file and returns the result tuple.
    Returns None if file is skipped or errors occur.
    """
    # 1. Safety check: Does file exist?
    try:
        file_size = os.path.getsize(filename)
    except OSError as e:
        print(f"Skipping {filename}: {e}", file=sys.stderr)
        return None

    # 2. Size check
    if file_size > max_size_bytes:
        return None
    
    # 3. Read & Hash
    try:
        with open(filename, "rb") as f:
            raw_bytes = f.read()
    except OSError as e:
        print(f"Error reading {filename}: {e}", file=sys.stderr)
        return None

    # Sanity check
    if file_size != len(raw_bytes):
        print(f"Warning: Size mismatch for {filename}", file=sys.stderr)
    
    h = hashlib.sha256(raw_bytes).hexdigest()

    # 4. ML Processing
    start_time = time.perf_counter()
    extractor = get_extractor()
    try:
        features = extractor.raw_features(raw_bytes)
        X = extractor.process_raw_features(features)
        model = get_model()
        score = float(model.predict([X])[0]) # type: ignore
    except Exception as e:
        print(f"Extraction/Inference error on {filename}: {e}", file=sys.stderr)
        score = -1.0 # Error sentinel
    
    duration = time.perf_counter() - start_time
    
    return filename, file_size, h, score, duration

def generate_targets(input_path, input_type, recursive, allowed_extensions):
    """
    Generator that yields valid full file paths to scan.
    Handles the logic for directories vs file-lists.
    """
    # Helper to check extensions
    def is_valid_ext(path):
        if not allowed_extensions:
            return True
        return path.lower().endswith(allowed_extensions)

    if input_type == "path-list":
        if not os.path.isfile(input_path):
            raise click.BadParameter(f"Path list file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            for line in f:
                path = line.strip().strip('\n') # Handle user whitespace and newlines
                if path:
                    # Recursively call this generator for the path in the list
                    # This allows a list to contain both files and directories
                    yield from generate_targets(path, "direct", recursive, allowed_extensions)

    elif input_type == "direct":
        abs_path = os.path.abspath(input_path)
        
        if os.path.isfile(abs_path):
            if is_valid_ext(abs_path):
                yield abs_path
                
        elif os.path.isdir(abs_path):
            for dirpath, dirnames, filenames in os.walk(abs_path, followlinks=True):
                # 1. Filter extensions first to save loops
                for filename in filenames:
                    if is_valid_ext(filename):
                        yield os.path.join(dirpath, filename)
                
                # 2. Handle recursion
                if not recursive:
                    dirnames[:] = []
        else:
            # Path doesn't exist
            print(f"Warning: Input path does not exist: {abs_path}", file=sys.stderr)

# --- CLI ---

@click.group()
def cli():
    pass

@cli.command(context_settings={'show_default': True})
@click.argument("input-path", type=str)
@click.option("--max-file-size-in-MB", type=int, default=100)
@click.option("--file-extensions", type=str, default="", help="a comma-separated list of extensions to look for, e.g. `.exe,.dll,.sys`; if empty, scan all files")
@click.option("--no-recursive", type=bool, is_flag=True, default=False, help="if specified, only the files right under the specified directory will be scanned (in case path-to-scan points to a directory)")
@click.option("--wait-time", type=float, default=1.0, help="time, in seconds, to wait after scanning each sample and before the next")
@click.option("-o", "--output", type=str, default="", help="path to output file; recursively create directories in that path if not exists; if empty, outputs to stdout")
@click.option("-i", "--input-type", type=click.Choice(['direct', 'path-list']), default="direct", help="""type of input-path, one of:

    - `direct`: input-path is the path to the file or directory to scan; or

    - `path-list`: input-path is the path to a file that contains a list of file/directory paths to scan, each on its own line.

""")
def scan(input_path, max_file_size_in_mb, file_extensions, no_recursive, wait_time, output, input_type):
    recursive = not no_recursive
    # Echo config
    print(f"Configuration:", file=sys.stderr)
    print(f"  Input: {input_path} ({input_type})", file=sys.stderr)
    print(f"  Output: {output if output else 'stdout'}", file=sys.stderr)
    print(f"  Max Size: {max_file_size_in_mb} MB", file=sys.stderr)
    print(f"  Extensions: {file_extensions if file_extensions else 'ALL'}", file=sys.stderr)
    print(f"  Recursive: {recursive}", file=sys.stderr)
    print(f"  Wait Time: {wait_time}s", file=sys.stderr)
    print(f"========================", file=sys.stderr)

    # Prepare constants
    max_size_bytes = max_file_size_in_mb * 1024 * 1024
    # Clean extensions tuple (empty tuple if string is empty)
    allowed_exts = tuple(x.strip().lower() for x in file_extensions.split(',') if x.strip())

    with get_output_stream(output) as output_file:
        writer = csv.writer(output_file, lineterminator='\n', quoting=csv.QUOTE_ALL)
        
        # WRITE HEADER
        # writer.writerow(["filename", "file_size", "sha256", "score", "scan_duration"])
        output_file.flush()

        # Main Loop
        count = 0
        for file_path in generate_targets(input_path, input_type, recursive, allowed_exts):
            
            result = process_single_file(file_path, max_size_bytes)
            
            if result:
                writer.writerow(result)
                output_file.flush()
                count += 1
            
            if wait_time > 0:
                time.sleep(wait_time)

    print(f"Done. Scanned {count} files.", file=sys.stderr)

if __name__ == "__main__":
    cli()
