

#### 基础训练
环境配置查看requirements.txt，数据准备，预训练权重可以从darknet官网下载。<br>
用yolov3训练自己的数据集，修改cfg，配置好data，用yolov3.weights初始化权重。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/yolov3.weights --epochs 100 --batch-size 32`

#### 稀疏训练
scale参数默认0.001，根据数据集，mAP,BN分布调整，数据分布广类别多的，或者稀疏时掉点厉害的适当调小s;-sr用于开启稀疏训练；--prune 0适用于prune.py，--prune 1 适用于其他剪枝策略。大的s一般稀疏较快但精度掉的快，小的s一般稀疏较慢但精度掉的慢；配合大学习率会稀疏加快，后期小学习率有助于精度回升。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1`

#### 通道剪枝策略一
由于yolov3中有五组共23处shortcut连接，对应的是add操作。<br>
<br>
`python prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --percent 0.85`



#### 层剪枝
针对每一个shortcut层前一个CBL进行评价，对各层的Gmma均值进行排序，取最小的进行层剪枝。为保证yolov3结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。是的，这里只考虑剪主干中的shortcut模块。但是yolov3中有23处shortcut，剪掉8个shortcut就是剪掉了24个层，剪掉16个shortcut就是剪掉了48个层，总共有69个层的剪层空间；实验中对简单的数据集剪掉了较多shortcut而精度降低很少。<br>
<br>
`python layer_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --shortcuts 12`

#### 微调finetune
剪枝的效果好不好首先还是要看稀疏情况，而不同的剪枝策略和阈值设置在剪枝后的效果表现也不一样，有时剪枝后模型精度甚至可能上升，而一般而言剪枝会损害模型精度，这时候需要对剪枝后的模型进行微调，让精度回升。<br>
<br>
`python train.py --cfg cfg/prune_0.85_my_cfg.cfg --data data/my_data.data --weights weights/prune_0.85_last.weights --epochs 100 --batch-size 32`

## License
Apache 2.0
