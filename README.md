# 多模态情感分析

## 代码目录说明

```
├── datasets     #　存放数据
|  └── data
|  └── train.txt
|  └── test_without_label.txt
|  └── dev.tsv 
|  └── train.tsv 
|  └── test.tsv
├── metrics　# 向量计算 
|  └── compute.py　　　
├── outputs              # 模型输出保存
|  └── pytorch_encoder.bin
|  └── pytorch_model.bin
|  └── test_without_label.txt #最终预测结果 
├── pre_trained_model　#保存的预训练模型
|  └── renet152.pth
├── processors　　　　　
|  └── util.py     # 辅助函数 
├── models　   # 模型代码
|  └── model.py
|  └── resnet.py

├──  get_data.py　#　数据处理 
├── lab05.ipynb   #　代码入口

├── run.py  
├── run.sh   #　运行图片+文件
├── run_image_only.sh   #　运行图片
├── run_text_only.sh   #　运行文字
├── run_test.sh   #　运行预测, 生成结果
```

## Requirements
```
pip install -r requirements.txt
```


## 运行

1. 下载预训练模型ResNet-152，并将该模型放入pre_trained_model文件下

   链接(https://download.pytorch.org/models/resnet152-b121ed2d.pth)

2. 进入lab05.ipynb, 依次运行cell即可



最后得出的结果在outputs文件夹下的 test_without_label.txt文件中



# 参考库和论文

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171-4186).
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

## github仓库

1. https://github.com/huggingface/transformers
2. https://github.com/pytorch/vision
3. https://github.com/KaimingHe/deep-residual-networks