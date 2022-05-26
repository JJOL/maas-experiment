# Setup a WSL2 instance: https://cloudbytes.dev/snippets/how-to-install-multiple-instances-of-ubuntu-in-wsl2
# Install CUDA in WSL2

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



# MT3 Dependency Setup
apt-get update -qq && apt-get install -qq fluidsynth build-essential libasound2-dev libjack-dev
# pip install --upgrade pip
# pip install nest-asyncio
# pip install pyfluidsynth

mkdir t5x_workspace
cd t5x_workspace
git clone --branch=main https://github.com/google-research/t5x
mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp
sed -i 's:jax\[tpu\]:jax:' setup.py