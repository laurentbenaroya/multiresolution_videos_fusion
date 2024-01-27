# goal
The goal is to demonstrate the use of good old laplacian pyramids for two videos frame by frame fusion with different coefficients for each channel (RGB) and each level of the pyramid. The fusion is done in the laplacian domain and the result is reconstructed in the spatial domain.
# code

## Create a virtual environment :
```
python3 -m venv myenv
source myenv/bin/activate
```
to deactivate the virtual environment :
`deactivate`

## Install requirements :
`pip install -r requirements.txt`

## Run program :

`python main.py --video1 video/Attal.mp4 --video2 video/Attal_wav2lip_gan.mp4 --output mytry --alphaR 0.3 0.1 0.2 0.3 0.8 --alphaB 0.5 0.5 0.5 0.6 0.7 --alphaG  0.4 0.9 0.2 0.01 1e-5`

## Display results

`vlc mytry/fused_video.mp4`

# credits
Laurent Benaroya, 2024
MIT LICENSE