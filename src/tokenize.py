print("Loading modules...")
from os import listdir
import os
import sys

# 1. Get midi dataset path
# 2. Each midi file convert it into token representation
# 3. Write each token representation to respective file as string token sequence:
# "x,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\n........"

tokenizer = Octuple()

def make_new_dir(target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

def transcribe(src_path, dest_path):
    print(f"Transcribing {src_path} to {dest_path}")
    midi = MidiFile(src_path)
    tokens = tokenizer.midi_to_tokens(midi)
    


def convert(src_path, dest_path):
    src_name = os.path.basename(os.path.normpath(src_path))
    if not os.path.isdir(src_path):
        print(f"Processing file {src_name}")
        if src_name[-4:] == ".wav":
            transcribe(src_path, dest_path)
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