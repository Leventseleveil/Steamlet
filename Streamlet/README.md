## 实时视频流应用平台

### 一、介绍

在当前已有的实时视频流传输应用平台上，主要分成两种类型：视频分块传输方式和分帧传输方式。视频分块传输方式是将多个视频帧进行编码封装后再进行网络传输，尽管存在一定的实时延迟但编码效率更高，便于实时视频流自适应传输播放。而视频分帧传输方式是将单个视频帧文件进行编码传输，能够满足超低时延要求，但其编码效率较低，同时也难以保证自适应流传输稳定性。鉴于本文实时视频流传输应用研究更多基于流自适应传输方案，因此采用视频分块传输方式来作为基础技术支撑。

另外，目前所存在的实时视频流传输研究平台主要包含三个缺陷点：

* 无法满足低延迟目标需求，由于单个视频分块大小一般为 2 到 3s，导致延迟较高；

* 不适合端侧网络模型部署，无法有效进行模型调用及状态信息交互；

* 缺少实时视频流会话仿真平台，导致网络模型在离线过程下难以有效训练。

基于以上问题和不足之处，我们设计并实现了一个基于端云协同的实时超分视频流传输平台 ECSRP-Streamlet，便于算法设计与研究应用平台的结合，以及最终实验效果的性能验证分析。

### 二、 项目启动

1. 安装核心依赖环境：
   * [install nodejs](http://nodejs.org/)
2. 安装项目依赖文件：
   * ```npm install```
3. 启动项目展示页面，运行视频流应用平台：
   * ```npm run start```
4. 打开平台页面，进行参数配置及实时视频流播放：
   * http://127.0.0.1:3000/samples/streamlet/testplayer.html

### 三、平台展示

1. 平台参数设置：在 ECSRP-Streamlet 平台应用中，提供了低延迟参数和算法逻辑策略的设置选项， 如下图所示。

   ![figure_4_3.png](https://s2.loli.net/2022/05/30/E5jpA2L7MHBS6sk.png)

2. 实时视频播放及指标输出：对于一个完整的实时视频流应用平台来说，视频播放器界面和各种参数指标显示才是该应用平台的核心，如下图所示。页面上方是视频内容生成和获取的 MPD 地址链接，通过点击加载按钮，向媒体服务器进行请求视频流初始加载和算法应用过程。页面左侧是视频播放器界面，视频画面内容会显示当前呈现的视频生成时间和所选视频码率级别。页面右上侧是一个当前时钟的相对时间，用于计算实时延迟值。页面右下侧是各种相关数据指标值，包括当前实时延迟、当前缓冲区占用率、视频分段下载的索引号、当前视频下载码率、最小延迟偏差、视频播放速率等数据指标。

   ![figure_4_4.png](https://s2.loli.net/2022/05/30/JqXjPFc6NlmkVL7.png)

3. 数据动态变化展示：为了更加直观显示视频数据指标的动态变化，这里采用了 Chart.js 作为动态图表展示框架，如下图所示。该动态数据变化曲线图分别描述了实时延迟、缓冲区占用率和视频播放速率数据指标的变化情况，每个数据指标值都是从实时视频流会话运行状态中进行采集得到的。通过观察该动态数据变化曲线，可以更加清晰明了地获取当前 ABR 算法下的实时视频流整体运行状态，同时有利于及时分析采用不同设置策略对于实时视频流的客观影响。比如，当设置不同网络吞吐率计算方式时，由于不同方法对于网络吞吐率预估有着较大的偏差，从而导致视频缓冲区变化也比较大。

   ![figure_4_5](C:\Users\libra\Desktop\研三工作文档\毕设资料\paper_word\figure\figure_new\平台展示\figure_4_5.png)

### 四、参考资料

1. https://dashif.org/
2. https://www.chartjs.org/
3. https://nodejs.org/en/
4. https://www.npmjs.com/