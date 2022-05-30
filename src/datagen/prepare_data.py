print("Loading modules...")
from os import listdir
import os
import sys

# 1. Get midi dataset path
# 2. Each midi file convert it into token representation
# 3. Write each token representation to respective file as string token sequence:
# "x,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\n........"


def make_new_dir(target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

def maxmin_stat(val, stats, feature):
    stats[f"max_{feature}"] = max(stats[f"max_{feature}"], val)
    stats[f"min_{feature}"] = min(stats[f"min_{feature}"], val)
    return stats

def calculate_stats(src_path):
    stats = {
        "max_pitch": 0,
        "min_pitch": 1000,
        "max_duration": 0,
        "min_duration": 1000,
        "max_velocity": 0,
        "min_velocity": 1000,
        "instrument_map": {},
        "max_instrument": 0
    }
    inst_counter = 0

    categories = listdir(src_path)
    for cat in categories:
        files = listdir(os.path.join(src_path, cat))
        for f in files:
            f = open(os.path.join(src_path, cat, f), "r")
            lines = f.readlines()
            for line in lines:
                elements = list(map(float, line.split(',')))
                stats = maxmin_stat(elements[0], stats, "pitch")
                stats = maxmin_stat(elements[1], stats, "velocity")
                stats = maxmin_stat(elements[2], stats, "duration")
                if int(elements[3]) not in stats["instrument_map"]:
                    stats["instrument_map"][int(elements[3])] = inst_counter
                    inst_counter += 1
    stats["max_instrument"] = inst_counter
    return stats

def normalize(val, stats, feature):
    max_val = stats[f"max_{feature}"]
    min_val = stats[f"min_{feature}"]
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val)

def vocab_index(val, stats, feature):
    binary_coded = format(stats[feature][val], '07b')
    formatted = ','.join(d for d in binary_coded)
    return formatted

def preprocess(src_path, dest_path, stats):
    print(f"Preprocessing {src_path} to {dest_path}")
    original = open(src_path, 'r')
    token_lines = original.readlines()

    out_lines = ""
    
    for token in token_lines:
        token = token.strip()
        elements = token.split(",")
        elements = list(map(float, elements))
        n_pitch = normalize(elements[0], stats, "pitch")
        n_vel = normalize(elements[1], stats, "velocity")
        n_dur = normalize(elements[2], stats, "duration")
        n_inst = vocab_index(elements[3], stats, "instrument_map")
        new_elements = [n_pitch, n_vel, n_dur, n_inst, elements[4], elements[5]]
        # new_token = ','.join(str(x) for x in new_elements)
        out_lines += f"{new_elements[0]:.6f},{new_elements[1]:.6f},{new_elements[2]:.6f},{new_elements[3]},{new_elements[4]:.6f},{new_elements[5]:.6f}\n"
    
    transformed = open(dest_path, 'w')
    transformed.writelines(out_lines)
    transformed.close()
    original.close()

def convert(src_path, dest_path, stats):
    src_name = os.path.basename(os.path.normpath(src_path))
    if not os.path.isdir(src_path):
        print(f"Processing file {src_name}")
        if src_name[-4:] == ".tok":
            preprocess(src_path, f"{dest_path[:-4]}.tok", stats)
    else:
        print(f"Processing dir {src_name}")
        make_new_dir(dest_path)
        files = listdir(src_path)
        for f in files:
            convert(f"{os.path.join(src_path, f)}",  f"{os.path.join(dest_path, f)}", stats)

def main(argv):
    if len(argv) < 3:
        print("Error: Missing inputs")
        return
    source_path = argv[1]
    dest_path = argv[2]

    source_name = os.path.basename(os.path.normpath(source_path))
    dest_name = os.path.basename(os.path.normpath(dest_path))

    print(f"Going to tokenize {source_name} ds into {dest_name} ds...")

    stats = calculate_stats(source_path)
    
    convert(source_path, dest_path, stats)

if __name__ == "__main__":
    main(sys.argv)