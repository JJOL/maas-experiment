print("Loading modules...")
from os import listdir
import os
import sys
import editdistance as ed

# 1. Get midi dataset path
# 2. Each midi file convert it into token representation
# 3. Write each token representation to respective file as string token sequence:
# "x,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\nx,x,x,x,x,x,x,x\n........"


def calculate_stats(src_path):
    stats = {
        "max_tokens": 0,
        "min_tokens": 1000,
        "avg_tokens": 0
    }
    token_total = 0
    n_samples = 0

    categories = listdir(src_path)
    for cat in categories:
        files = listdir(os.path.join(src_path, cat))
        for f in files:
            f = open(os.path.join(src_path, cat, f), "r")
            tokens = len(f.readlines())
            if tokens > stats["max_tokens"]:
                stats["max_tokens"] = tokens
            if tokens < stats["min_tokens"]:
                stats["min_tokens"] = tokens
            token_total += tokens

            n_samples += 1
    stats["avg_tokens"] = token_total / n_samples
    return stats

def file_tokens(content: str):
    lines = content.split('\n')
    lines = lines[0:800]
    tokens = []
    for l in lines:
        for x in l.split(','):
            tokens.append(x)
    return tokens

def token_editdistance(fileA, fileB):
    fA = open(fileA, 'r')
    fA_tokens = file_tokens(fA.read())
    fA.close()
    fB = open(fileB, 'r')
    fB_tokens = file_tokens(fB.read())
    fB.close()

    return ed.eval(fA_tokens, fB_tokens)

cached_distances = {}

def pair_key(strA, strB):
    return f"{strA}-{strB}" if strA < strB else f"{strB}-{strA}"

def convert(src_path, stats):
    categories = listdir(src_path)
    ds_dist = 0
    for cat in categories:
        print(f"Calculating {cat}...")
        cat_dist = 0
        class_samples = listdir(f"{os.path.join(src_path, cat)}")
        for sample in class_samples:
            for other_sample in class_samples:
                if sample != other_sample:
                    if pair_key(sample, other_sample) not in cached_distances:
                        dist = token_editdistance(os.path.join(src_path, cat, sample), os.path.join(src_path, cat, other_sample))
                        cached_distances[pair_key(sample, other_sample)] = dist
                    else:
                        dist = cached_distances[pair_key(sample, other_sample)]
                    cat_dist += dist
        cat_dist /= len(class_samples)
        print(f"Category '{cat}' Avg Token Edit Distance: {cat_dist}")
        ds_dist += cat_dist
    ds_dist /= len(categories)
    print(f"Data Set Avg Token Edit Distance: {ds_dist}")

def main(argv):
    if len(argv) < 2:
        print("Error: Missing inputs")
        return
    source_path = argv[1]

    stats = calculate_stats(source_path)
    
    convert(source_path, stats)

if __name__ == "__main__":
    main(sys.argv)