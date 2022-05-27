unzip /content/covers80.zip
python src/dataget/transcribe.py training/covers80 training/covers80mt3_midi mt3
zip -r "covers80mt3_midi.zip" training/covers80mt3_midi