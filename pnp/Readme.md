1. create env
```
conda activate cpnp python=3.9 -y
conda activate cpnp
```

2. install pytorch
```
# More installation methods can be found at https://pytorch.org/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

3. install dependence
```
pip install -r requirement.txt
```

4. get the cpnp repo
```
git clone -b cpnp1 https://github.com/WinSanyu/cpnp.git
```

5. download SNDnCNN checkpoint
download checkpoint from

链接: https://pan.baidu.com/s/1zgclkWipLdC6lt_9ohCohw?pwd=fe2m 提取码: fe2m 复制这段内容后打开百度网盘手机App，操作更方便哦


6. run cpnp1
```
cd cpnp/pnp
# 5%
python eval_cpnp.py -opt ../options/pnp/pnp_sndncnn5.json
# 10%
python eval_cpnp.py -opt ../options/pnp/pnp_sndncnn10.json
```