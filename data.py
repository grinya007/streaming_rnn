import os
import re
import tarfile
from pathlib import Path

def read_files_dir(input_dir):
    path = Path(input_dir)
    if not path.is_dir():
        return ''
    for text_file in sorted(path.iterdir()):
        if text_file.name.startswith('.'):
            continue
        with text_file.open() as f:
            while f.tell() < os.fstat(f.fileno()).st_size:
                line = f.readline()
                yield line

def read_files_tar(tar_file):
    tar = tarfile.open(tar_file, 'r:gz')
    members = tar.getmembers()
    members.sort(key=lambda m: m.name)
    for member in members:
        f = tar.extractfile(member)
        if f is not None:
            yield f.read().decode('utf8')

def strip_words(text):
    prev_word = None
    detached_suffixes = set(["'s", "'d", "'ve", "'m"])
    for match in re.finditer(r"[-'’a-zA-Z0-9]+|[\.\,\?\!\:\;]", text):
        words = match.group(0).lower()
        if words == 'br':
            continue
        for word in re.split(r'-{2,}', words):
            if word in detached_suffixes:
                if prev_word is not None:
                    yield prev_word + word
                    prev_word = None
            else:
                if prev_word is not None:
                    yield prev_word
                prev_word = word
    if prev_word is not None:
        yield prev_word
