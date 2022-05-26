#!/bin/bash
DS_DOWNLOADER=$1
DS_TRANSCRIPTION=$2
DS_TOKENIZER=$3
DS_NAME=$4

echo Dataset Generation Input Parameters:
echo - DS Name: $DS_NAME
echo - Download Script: $DS_DOWNLOADER
echo - Transcription Method: $DS_TRANSCRIPTION
echo - Tokenizer: $DS_TOKENIZER

# 1. Download and normalize raw data
echo "./src/datagen/$DS_DOWNLOADER"

# 2. Transcribe data
python src/datagen/transcribe.py "training/$DS_NAME" "training/${DS_NAME}_midi"

# 3. Tokenize data
python src/datagen/mytokenizer.py "training/${DS_NAME}_midi" "training/${DS_NAME}_token"