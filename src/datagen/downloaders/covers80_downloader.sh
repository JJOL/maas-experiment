
if [[ ! -d training/covers80 ]]
then

    # 1. Get Music Files
    cp -r /home/jjolme/Tesina/youtube-downloader/downloaded training/covers80

    # 2. Convert from MP3 to WAV
    for song in $(ls training/covers80);
    do
        echo "Processing Flder $song..."
        for i in $(ls training/covers80/$song);
        do
            echo "Attempting convert song $i..."
            ffmpeg -i "training/covers80/$song/$i" -acodec pcm_u8 -ar 22050 "training/covers80/$song/${i%.*}.wav"
            rm "training/covers80/$song/$i"
        done    
    done

fi