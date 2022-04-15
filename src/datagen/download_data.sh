# 1. make dir training for data and model
mkdir -p training
mkdir -p training/assets

# 2. Download Zip File
if [[ ! -f training/assets/gtzan.tar.gz ]]
then
    wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -O training/assets/gtzan.tar.gz
fi

# 3. Unzip into GTZAN
if [[ ! -d training/gtzan ]]
then
    mkdir -p training/gtzan
    tar -xzvf training/assets/gtzan.tar.gz -C training/gtzan
fi