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
        yield text_file.read_text()

def read_files_tar(tar_file):
    tar = tarfile.open(tar_file, 'r:gz')
    members = tar.getmembers()
    members.sort(key=lambda m: m.name)
    for member in members:
        f = tar.extractfile(member)
        if f is not None:
            yield f.read().decode('utf8')

def strip_words(text):
    ticks = set(['’', '’’', "'", "''"])
    for match in re.finditer(r"[-'’a-zA-Z0-9\$]+|[\—\.\,\?\!\:\;]", text):
        word = match.group(0).lower()
        if word in ticks:
            continue
        yield word
