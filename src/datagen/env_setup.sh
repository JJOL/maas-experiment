## OMINZART Installation followed as official guide: https://music-and-culture-technology-lab.github.io/omnizart-doc/quick-start.html

# Install the prerequisites manually since there are some dependencies can't be
# resolved automatically.
pip install numpy Cython
pip install wheel # PENDING TO CHECK IF REALLY NECESSARY

# Additional system packages are required to fully use Omnizart.
sudo apt-get install libsndfile-dev fluidsynth ffmpeg

# Install Omnizart
pip install omnizart

# Then download the checkpoints
omnizart download-checkpoints

# Additional Updates
pip install numba



## Miditok Installation followed as official guide: https://github.com/Natooz/MidiTok
pip install miditok
pip install matplotlib