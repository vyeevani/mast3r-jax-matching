conda create --prefix ./env python=3.11 cmake

./env/bin/pip install torch torchvision jax equinox jaxtyping einops 

cd mast3r
../env/bin/pip install -r requirements.txt

cd dust3r
../env/bin/pip install -r requirements.txt
cd ..

cd mast3r
../env/bin/python setup.py install
cd ..

cd dust3r
../env/bin/python setup.py install
cd ..

cd mast3r
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
cd ..
