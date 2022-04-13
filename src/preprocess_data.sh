# 1. Transform to MIDI audio dataset
if [[ ! -d training/gtzan_midi ]]
then
    python preprocess.py training/gtzan training/gtzan_midi
fi

# 2. Transform to Tokens MIDI dataset
if [[ ! -d training/gtzan_token ]]
then
    python tokenize.py training/gtzan_midi training/gtzan_token
fi