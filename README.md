
``` 
docker run --gpus all -it -v $PWD:/work --workdir /work --name ganrl nvcr.io/nvidia/pytorch:21.09-py3   
```


``` 
conda init
source ~/.bashrc
conda create -n ganrl --clone base
conda activate ganrl

apt-get install -y git 
pip install git+https://github.com/koyaishikawa/kirl
pip install git+https://github.com/koyaishikawa/kifin
```