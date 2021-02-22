动态形状是一种延迟指定某个或全部张量维度知道运行时的能力，下面是构建一个动态形状引擎的步骤：

1. 网络定义必须不能有隐式的输入batch维度，通过下面的代码构建`INetworkDefinition`：

   ```cpp
   IBuilder::createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
   ```

2. 使用`-1`指定输入张量的每个运行时维度作为这个维度的占位符；

3. 在构建时指定一个或多个优化概要（optimization profiles）来确定运行时维度输入的维度所允许的范围，而这些维度是自动调试器（autotuner）需要优化的；

4. 要使用引擎，必须：

   a. 从引擎像没有动态输入那样创建一个执行上下文；

   b. 指定步骤3中的一个优化概要来覆盖输入维度；

   c. 指定执行上下文的输入维度，

