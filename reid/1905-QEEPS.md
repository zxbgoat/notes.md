行人搜索近来作为一项在一些未裁剪图像中找到提供为裁剪标本人物的任务浮现，其挑战在于搜索图像包含混乱背景和遮挡，而查询人物可能以不同的视角、姿势、大小和光照出现。典型的方法会将这个任务分为人物定位（检测）和重识别两个任务，并通过不同的监督网络循序解决每个任务，但这会丢失对重识别网络有用的上下文信息，而检测任务也无法从查询挖掘信息因其是在重识别之前单独运行。

而人在一张图像搜索人物时，不仅会查看每个个体，也会搜寻特定的模式作为额外的线索。受此启发，我们引入端到端查询导引的人物搜索（Query-guided End-to-End Person Search，QEEPS），同时优化检测和重识别，并使这两方面都以查询图像为条件。

我们受OIM启发，设计一个由重识别分支包围的检测器。正如在OIM中，我们同时采用OIM损失函数来优化网络。我们的模型另外配备了：

- 一个查询引导的孪生SE（Squeeze-and-Extraction）网络（QSSE-Net），它扩展SE技术用查询（query）和搜索（gallery）通道的（全局，图像层次）相互依赖来重校准逐通道孪生特征响应；
- 一个查询导引的RPN（QRPN），它补充了查询特定提议的并行RPN，应用了一个修改的SE块来强调（除特征通道外）特殊的空间特征，尤其提出了查询特定的判别特征；
- 一个查询相似度网络（QSimNet），它取查询和搜索图像提议的重识别特征并提供一个查询导引的重识别分数，当添加到基准时，单独的QSimNet就在CUHK-SYSU数据集上（图像搜索库大小为100）改善了多达7.1pp的mAP和4.33pp的CMC top-1。

所有这些一起，QEEPS在CUHK-SYSU数据集上设立了0.889mAP和0.891top1的新标杆；而在PRW数据集上也达到了0.371mAP和0.767top1的新高度。