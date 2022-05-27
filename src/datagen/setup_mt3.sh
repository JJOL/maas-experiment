apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev
pip install --upgrade pip
pip install nest-asyncio
pip install pyfluidsynth

mkdir mt3_workspace
cd mt3_workspace

# Install manually CLU of latest known working commit
git clone --branch=main https://github.com/google/CommonLoopUtils
cd CommonLoopUtils
git checkout 84b777c42dfd3fb6685537138433bfeb5241a006
cd ..
mv CommonLoopUtils CommonLoopUtils_tmp; mv CommonLoopUtils_tmp/* .; rm -r CommonLoopUtils_tmp
python3 -m pip install -e .

# install t5x
git clone --branch=main https://github.com/google-research/t5x
# Change to latest known working commit
cd t5x
git checkout 16649d7365ce8586023cb3b489bea277420f623e
cd ..

mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp
sed -i 's:jax\[tpu\]:jax:' setup.py
# Remove CLU dependency as it has been manually installed
sed -i '/CommonLoopUtils/d' setup.py
python3 -m pip install -e .

# install mt3
# Necessary installation
pip install llvmlite --ignore-installed
git clone --branch=main https://github.com/magenta/mt3
mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
python3 -m pip install -e .

# copy checkpoints
gsutil -q -m cp -r gs://mt3/checkpoints .