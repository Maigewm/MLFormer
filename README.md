# MLFormer
Code for the paper "MLFormer: Unleashing Efficiency without Attention for Multimodal Knowledge Graph Embedding"
# Requirements

  pip install -r requirements.txt
  
# Data
  *FB15k-237:
  To obtain the image data for FB15k-237, you can access the [mmkb](https://github.com/mniepert/mmkb) repository, which offers a list of image URLs associated with the dataset. For more information on entity descriptions, you can refer to the [kg-bert](https://github.com/yao8839836/kg-bert) repository.
  *WN18:
  If you want to obtain entity images in WN18, you can use ImageNet as a source. The detailed steps for doing so can be found in the [RSME](https://github.com/wangmengsd/RSME) repository. The process outlined in this repository provides a guide for associating ImageNet image URLs with entities in the WN18 dataset.
  
# Run
  Firstly, pretrain the model through MLM task:
    ```
    bash scripts/pretrain_taskname.sh
    ```
  Then train the model from checkpoint:
   ```
    bash scripts/taskname.sh
   ```
# Acknowledgement
  The code is based on [MKGformer](https://github.com/zjunlp/MKGformer). We thank the authors for sharing their codes sincerely.
