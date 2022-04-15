print("Loading modules...")
from os import listdir
import os
import sys

from miditok import Octuple
from miditoolkit import MidiFile

# 1. Get midi dataset path
# 2. Each midi file convert it into token representation
# 3. Write each token representation to respective file as string token sequence:
# "x,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\n........"

tokenizer = Octuple()
max_len = -1
max_len_name = ""
min_len = 10000
min_len_name = ""
avg_len = 0
count = 0
len_sum = 0

def make_new_dir(target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

def tokenize(src_path, dest_path):
    print(f"Tokenizing {src_path} to {dest_path}")
    midi = MidiFile(src_path)
    tokens = tokenizer.midi_to_tokens(midi)
    token_len = len(tokens)

    global count
    global len_sum
    global max_len
    global max_len_name
    global min_len
    global min_len_name
    count += 1
    len_sum += token_len
    if token_len > max_len:
        max_len = token_len
        max_len_name = src_path
    if token_len < min_len:
        min_len = token_len
        min_len_name = src_path

    if count > 200:
        quit()
    # print(f"File {src_path} has {len(tokens)}!")

    with open(dest_path, mode="w") as out_file:
        for t in tokens:
            out_file.write(f"{t[0]},{t[1]},{t[2]},{t[3]},{t[4]},{t[5]}\n")
            

def convert(src_path, dest_path):
    src_name = os.path.basename(os.path.normpath(src_path))
    if not os.path.isdir(src_path):
        print(f"Processing file {src_name}")
        if src_name[-4:] == ".mid":
            tokenize(src_path, f"{dest_path[:-4]}.tok")
    else:
        print(f"Processing dir {src_name}")
        make_new_dir(dest_path)
        files = listdir(src_path)
        for f in files:
            convert(f"{os.path.join(src_path, f)}",  f"{os.path.join(dest_path, f)}")

def main(argv):
    if len(argv) < 3:
        print("Error: Missing inputs")
        return
    source_path = argv[1]
    dest_path = argv[2]

    source_name = os.path.basename(os.path.normpath(source_path))
    dest_name = os.path.basename(os.path.normpath(dest_path))

    print(f"Going to tokenize {source_name} ds into {dest_name} ds...")
    
    convert(source_path, dest_path)

if __name__ == "__main__":
    main(sys.argv)

    avg_len = len_sum / count
    print(f"Max Token Len: {max_len} / {max_len_name}")
    print(f"Min Token Len: {min_len} / {min_len_name}")
    print(f"Avg Token Len: {avg_len} from {count} samples")