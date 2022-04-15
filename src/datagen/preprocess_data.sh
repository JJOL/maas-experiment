# 1. Transform to MIDI audio dataset
# if [[ ! -d training/gtzan_midi ]]
# then
    python src/transcribe.py training/gtzan training/gtzan_midi
# fi

# 2. Transform to Tokens MIDI dataset
# if [[ ! -d training/gtzan_token ]]
# then
#     # python src/tokenize.py training/gtzan_midi training/gtzan_token
# fi