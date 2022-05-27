print("Loading modules...")
from os import listdir
import os
import sys

def make_new_dir(target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

def transcribe(src_path, dest_path):
    from omnizart.music import app
    print(f"Transcribing {src_path} to {dest_path}")
    app.transcribe(src_path, 
        # model_path="/mnt/d/Courses/Tesina/other_env/env/lib/python3.8/site-packages/omnizart/checkpoints/music/music_piano",
        # model_path="/home/jjolme/Tesina/maas-experiment/env/lib/python3.8/site-packages/omnizart/checkpoints/music/music_piano",
        model_path="/home/jjolme/Tesina/maas-experiment/env/lib/python3.8/site-packages/omnizart/checkpoints/music/music_piano",
        output=dest_path)

def transcribe_mt3(src_path, dest_path):
    import mt3_lib as mt3
    print(f"Transcribing {src_path} to {dest_path} with MT3")
    mt3.transcribe(src_path, "/content/mt3_workspace/", output=dest_path)

def convert(src_path, dest_path, method):
    src_name = os.path.basename(os.path.normpath(src_path))
    if not os.path.isdir(src_path):
        print(f"Processing file {src_name}")
        if src_name[-4:] == ".wav":
            if os.path.exists(dest_path):
                print("File already processed! Ignoring...")
                return
            if method == "mt3":
                transcribe_mt3(src_path, dest_path)
            else:
                transcribe(src_path, dest_path)
    else:
        print(f"Processing dir {src_name}")
        make_new_dir(dest_path)
        files = listdir(src_path)
        for f in files:
            convert(f"{os.path.join(src_path, f)}",  f"{os.path.join(dest_path, f)}", method)


def main(argv):
    if len(argv) < 3:
        print("Error: Missing inputs")
        return
    source_path = argv[1]
    dest_path = argv[2]
    method = argv[3]

    source_name = os.path.basename(os.path.normpath(source_path))
    dest_name = os.path.basename(os.path.normpath(dest_path))

    print(f"Going to transform {source_name} ds into {dest_name} ds...")
    
    convert(source_path, dest_path, method)
    

    # for file in files:
    #     print(f"Processing {file}...")
    #     fpath = f"GTZAN/genres/disco/{file}"
    #     midi = app.transcribe(f"{fpath}", model_path="/mnt/d/Courses/Tesina/other_env/env/lib/python3.8/site-packages/omnizart/checkpoints/music/music_piano", output="GTZAN_MIDI/genres/disco")


if __name__ == "__main__":
    main(sys.argv)