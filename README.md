# BYTETrack
BYTETrack: Multi-Object Tracking BY Associating Every DeTEction Box

<summary>Installation</summary>

Step1. Install BYTETrack.
```shell
git clone https://github.com/ifzhang/BYTETrack.git
cd BYTETrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [apex](https://github.com/NVIDIA/apex).

```shell
# skip this step if you don't want to train model.
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step3. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step4. Others
```shell
pip3 install cython_bbox
```

<summary>Prepare datasets</summary>

Prepare coco format mot dataset.
```shell
cd <BYTETrack_HOME>
mkdir datasets
ln -s /path/to/your/mot ./datasets/mot
```
Change 'data_dir' in get_eval_loader() in exps/example/mot/yolox_x_ch.py ("mot" to "dancetrack")
```
data_dir=os.path.join(get_yolox_datadir(), "mot"),
```


<summary>Prepare pretrained models</summary>

```shell
cd <BYTETrack_HOME>
mkdir pretrained
cd pretrained
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/zhangyifu/debug1/models/bytetrack_models.tar.gz
tar -zxvf bytetrack_models.tar.gz
```

<summary>Run tracking</summary>

Run BYTETrack:

```shell
cd <BYTETrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
```

Run other trackers:
```shell
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ch_150.py -c pretrained/yolox_x_ch_150.pth.tar -b 1 -d 1 --fp16 --fuse
```

<summary>Train on CrowdHuman</summary>
Mix crowdhuman_train and crowdhuman_val and put the crowdhuman folder under datasets.

```shell
cd <BYTETrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_ch_150.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

