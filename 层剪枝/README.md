# yolov3-channel-and-layer-pruning
本项目以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为基础实现，根据论文[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)原理基于bn层Gmma系数进行通道剪枝，下面引用了几种不同的通道剪枝策略，并对原策略进行了改进，提高了剪枝率和精度；在这些工作基础上，又衍生出了层剪枝，本身通道剪枝已经大大减小了模型参数和计算量，降低了模型对资源的占用，而层剪枝可以进一步减小了计算量，并大大提高了模型推理速度；通过层剪枝和通道剪枝结合，可以压缩模型的深度和宽度，某种意义上实现了针对不同数据集的小模型搜索。<br>
<br>
项目的基本工作流程是，使用yolov3训练自己数据集，达到理想精度后进行稀疏训练，稀疏训练是重中之重，对需要剪枝的层对应的bn gamma系数进行大幅压缩，理想的压缩情况如下图，然后就可以对不重要的通道或者层进行剪枝，剪枝后可以对模型进行微调恢复精度，后续会写篇博客记录一些实验过程及调参经验，在此感谢[行云大佬](https://github.com/zbyuan)的讨论和合作！<br>
<br>
![稀疏](https://github.com/tanluren/yolov3-channel-and-layer-pruning/blob/master/data/img/1.jpg)

<br>




#### 基础训练
环境配置查看requirements.txt，数据准备参考[这里](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)，预训练权重可以从darknet官网下载。<br>
用yolov3训练自己的数据集，修改cfg，配置好data，用yolov3.weights初始化权重。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/yolov3.weights --epochs 100 --batch-size 32`

#### 稀疏训练
scale参数默认0.001，根据数据集，mAP,BN分布调整，数据分布广类别多的，或者稀疏时掉点厉害的适当调小s;-sr用于开启稀疏训练；--prune 0适用于prune.py，--prune 1 适用于其他剪枝策略。稀疏训练就是精度和稀疏度的博弈过程，如何寻找好的策略让稀疏后的模型保持高精度同时实现高稀疏度是值得研究的问题，大的s一般稀疏较快但精度掉的快，小的s一般稀疏较慢但精度掉的慢；配合大学习率会稀疏加快，后期小学习率有助于精度回升。<br>
注意：训练保存的pt权重包含epoch信息，可通过`python -c "from models import *; convert('cfg/yolov3.cfg', 'weights/last.pt')"`转换为darknet weights去除掉epoch信息，使用darknet weights从epoch 0开始稀疏训练。<br>
<br>
`python train.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1`
* ##### 稀疏策略一：恒定s
这是一开始的策略，也是默认的策略。在整个稀疏过程中，始终以恒定的s给模型添加额外的梯度，因为力度比较均匀，往往压缩度较高。但稀疏过程是个博弈过程，我们不仅想要较高的压缩度，也想要在学习率下降后恢复足够的精度，不同的s最后稀疏结果也不同，想要找到合适的s往往需要较高的时间成本。<br>
<br>
`bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))`
* ##### 稀疏策略二：全局s衰减
关键代码是下面这句，在epochs的0.5阶段s衰减100倍。前提是0.5之前权重已经完成大幅压缩，这时对s衰减有助于精度快速回升，但是相应的bn会出现一定膨胀，降低压缩度，有利有弊，可以说是牺牲较大的压缩度换取较高的精度，同时减少寻找s的时间成本。当然这个0.5和100可以自己调整。注意也不能为了在前半部分加快压缩bn而大大提高s，过大的s会导致模型精度下降厉害，且s衰减后也无法恢复。如果想使用这个策略，可以在prune_utils.py中的BNOptimizer把下面这句取消注释。<br>
<br>
`# s = s if epoch <= opt.epochs * 0.5 else s * 0.01`
* ##### 稀疏策略三：局部s衰减
关键代码是下面两句，在epochs的0.5阶段开始对85%的通道保持原力度压缩，15%的通道进行s衰减100倍。这个85%是个先验知识，是由策略一稀疏后尝试剪通道几乎不掉点的最大比例，几乎不掉点指的是相对稀疏后精度；如果微调后还是不及baseline，或者说达不到精度要求，就可以使用策略三进行局部s衰减，从中间开始重新稀疏，这可以在牺牲较小压缩度情况下提高较大精度。如果想使用这个策略可以在train.py中把下面这两句取消注释，并根据自己策略一情况把0.85改为自己的比例，还有0.5和100也是可调的。策略二和三不建议一起用，除非你想做组合策略。<br>
<br>
`#if opt.sr and opt.prune==1 and epoch > opt.epochs * 0.5:`<br>
`#  idx2mask = get_mask2(model, prune_idx, 0.85)`

#### 通道剪枝策略一
策略源自[Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)，这是一种保守的策略，因为yolov3中有五组共23处shortcut连接，对应的是add操作，通道剪枝后如何保证shortcut的两个输入维度一致，这是必须考虑的问题。而Lam1360/YOLOv3-model-pruning对shortcut直连的层不进行剪枝，避免了维度处理问题，但它同样实现了较高剪枝率，对模型参数的减小有很大帮助。虽然它剪枝率最低，但是它对剪枝各细节的处理非常优雅，后面的代码也较多参考了原始项目。在本项目中还更改了它的阈值规则，可以设置更高的剪枝阈值。<br>
<br>
`python prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --percent 0.85`



#### 层剪枝
这个策略是在之前的通道剪枝策略基础上衍生出来的，针对每一个shortcut层前一个CBL进行评价，对各层的Gmma均值进行排序，取最小的进行层剪枝。为保证yolov3结构完整，这里每剪一个shortcut结构，会同时剪掉一个shortcut层和它前面的两个卷积层。是的，这里只考虑剪主干中的shortcut模块。但是yolov3中有23处shortcut，剪掉8个shortcut就是剪掉了24个层，剪掉16个shortcut就是剪掉了48个层，总共有69个层的剪层空间；实验中对简单的数据集剪掉了较多shortcut而精度降低很少。<br>
<br>
`python layer_prune.py --cfg cfg/my_cfg.cfg --data data/my_data.data --weights weights/last.pt --shortcuts 12`

#### 微调finetune
剪枝的效果好不好首先还是要看稀疏情况，而不同的剪枝策略和阈值设置在剪枝后的效果表现也不一样，有时剪枝后模型精度甚至可能上升，而一般而言剪枝会损害模型精度，这时候需要对剪枝后的模型进行微调，让精度回升。训练代码中默认了前6个epoch进行warmup，这对微调有好处，有需要的可以自行调整超参学习率。<br>
<br>
`python train.py --cfg cfg/prune_0.85_my_cfg.cfg --data data/my_data.data --weights weights/prune_0.85_last.weights --epochs 100 --batch-size 32`

## License
Apache 2.0
