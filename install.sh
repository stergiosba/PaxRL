DEVICE=$1
echo "No device was specified, defaulting to cpu"
if [ -z $DEVICE ]; then
    DEVICE="cpu"
else
    DEVICE=$1
fi
echo "Installing PaxRL with device: $DEVICE for JAX"
pip install -r requirements.txt
if [ $DEVICE == "cpu" ]; then
    pip install --upgrade "jax[cpu]"

elif [ $DEVICE == "gpu" ]; then
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi