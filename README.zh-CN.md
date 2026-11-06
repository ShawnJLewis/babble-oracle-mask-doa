# 五种麦克风阵列上的 oracle 掩码 DOA 上界

本仓库在混响房间里合成扩散 babble，用直达声信号算出 oracle 时频掩码，再把掩码送进经典波达方向（DOA, direction of arrival）估计器。每种阵列回答一个问题：

> 当掩码完全准确、babble 又是扩散场时，时频掩码最多能把经典 DOA 估计器推到什么水平？

得到的数字是**上界**。掩码用到了直达声信号，真实系统拿不到，所以这些数字说明掩码前端最好能做到哪一步，不代表任何已部署系统的实测性能。

处理链路：

```
CMU ARCTIC wav  ->  镜像源 RIR  ->  混响目标语音 + 扩散 babble
                                    |
                                    +-> 混合信号、直达声分量
                                              |
                                 oracle IRM / PSM 掩码
                                              |
                    GCC-PHAT | MGCC | SR-SNR | SV | MUSIC
                                              |
                                角度估计 -> ACC、MAE
```

全流程不含训练。共注册五种阵列，从双麦参照到非共面的车载座舱。

## 安装

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

需要 Python 3.10 及以上，依赖 `numpy`、`scipy`、`soundfile`、`pyroomacoustics`。全部在 CPU 上运行，单核跑完 a2 的全量评测只需几分钟。

不想安装时，可以直接在源码树里用 `python -m oracle_doa` 运行，把 `PYTHONPATH` 指向 `src`（PowerShell 里写 `$env:PYTHONPATH="src"`）即可找到包。

## 快速开始

```bash
oracle-doa download                    # 下载 CMU ARCTIC，约 50 MB、460 个文件
oracle-doa rirs --array a2             # 生成 RIR 缓存，耗时较长，见「开销」一节
oracle-doa run --array a2 --utt 200    # 评测，结果写入 results/a2_oracle.json
```

缓存生成之后，只跑一个条件很快：

```bash
oracle-doa run --array a2 --t60 0.2 --utt 20 --mask psm --backend mgcc,sr_snr
```

```
[a2] 2 mics, 1 conditions, 20 utterances, masks ('psm',)
  d1.5_t0.20_s-6: 37 talkers, best masked ACC@tol 100.0% (20 utt)
```

`--array all` 按顺序跑完五种阵列。`oracle-doa arrays` 打印下一节的几何表。

## 阵列

| 代号 | 阵元数 | 几何 | 最小间距 | 孔径 | 无缠绕 `f < c/2d` | 房间（m） | 距离 | T60 网格（s） | 容差 |
|---|---|---|---|---|---|---|---|---|---|
| `a2` | 2 | 沿 x 轴直线，±0.1 m | 0.20 m | 0.20 m | < 858 Hz | 8×8×3 | 1.5 m | 0, 0.2, 0.4, 0.6, 0.8, 1.0 | 5° |
| `b8` | 8 | 直线，MIRD 4-4-4-8-4-4-4 cm | 0.04 m | 0.32 m | < 4288 Hz | 6×6×2.4 | 1.0、2.0 m | 0.16, 0.36, 0.61 | 7.5° |
| `spk6` | 6 | 环阵，半径 45 mm，位于顶棚高度 | 0.045 m | 0.09 m | < 3811 Hz | 6×6×2.4 | 1.0、2.0 m | 0.16, 0.36, 0.61 | 7.5° |
| `rob5` | 5 | 环阵，半径 110 mm，低于声源 1.15 m | 0.129 m | 0.209 m | < 1326 Hz | 6×6×2.4 | 1.0、2.0 m | 0.16, 0.36, 0.61 | 7.5° |
| `car4` | 4 | 非共面，两个 A 柱加两个顶灯 | 0.600 m | 2.00 m | < 286 Hz | 2.8×1.6×1.4 | 0.8 m | 0.16, 0.36, 0.61 | 10° |

方向搜索只覆盖前半平面，-90° 到 +90°，步长 1°。直线阵与环阵分不清声源和它在背后的镜像，因此后半平面的方向不在评测范围内。

无缠绕带宽影响很大：`car4` 的 257 个频点里只有约 9 个不发生相位缠绕，`a2` 约 27 个，`spk6` 约 122 个。`car4` 的 10° 容差来自这个限制，不表示判据放宽。

## 条件与 babble

| 代号 | 条件数 | 生成方式 | babble talker 数 |
|---|---|---|---|
| `a2` | 6 | 1 个距离 × 6 个 T60 | 37（1.5 m 处 5° 网格） |
| `b8` | 6 | 2 个距离 × 3 个 T60 | 26（2 个距离 × 13 个方向） |
| `spk6` | 6 | 2 个距离 × 3 个 T60 | 26 |
| `rob5` | 6 | 2 个距离 × 3 个 T60 | 26 |
| `car4` | 3 | 1 个距离 × 3 个 T60 | 13 |

每个干扰位置各配一位说话人，babble 因此呈扩散场而非单一相干干扰。每个条件内按固定 babble 信噪比取样，默认 -6 dB（`--snr` 可改），信噪比以混响后的目标语音为信号计算。

RIR 由镜像源法生成，墙面吸收由 Sabine 关系 `alpha = 0.161 V / (T60 S)` 设定，T60 因此是受控输入而非实测输出。直达径单独用 0 阶镜像源计算，oracle 掩码才能算准。

## Oracle 掩码

对每个麦克风分别计算：

```
IRM = |Yd|^2 / (|Yd|^2 + |Ymix - Yd|^2)
PSM = clip( IRM * cos( angle(Ymix) - angle(Yd) ), 0, 1 )
```

`Yd` 是直达声目标语的 STFT，`Ymix` 是混合信号的 STFT。IRM 的分母同时含晚期混响与 babble，所以这是**直达声主导**掩码，并非语音对噪声的掩码。后文引用的论文第 8 节用的是同一构造。

掩码逐麦克风独立计算，后端按 `M_p * M_q` 合并到麦克风对上。

## 后端

| 名称 | 是否用掩码 | 统计量 | 说明 |
|---|---|---|---|
| `gcc_phat` | 否 | 通道间相位，权重恒为 1 | 参照，不用掩码 |
| `mgcc` | 是 | 以 `M_p M_q` 加权的通道间相位 | 掩码加权 GCC-PHAT |
| `sr_snr` | 是 | 每对的 MVDR 输出信噪比，信号与噪声协方差分别由 `M` 与 `(1-M)` 得到 | 导向响应信噪比 |
| `sv` | 是 | 每个 2×2 掩码加权协方差的主特征向量相位 | 导向矢量法 |
| `music` | 可选 | 互补协方差的噪声子空间 | 逐频带归一后求和 |

所有估计器都用一次 `(pairs, directions, bins)` 上的 einsum 给整个方向网格打分，代价不随候选方向数增长。导向相位只由麦克风坐标与声源距离算出，搜索过程不含任何学习量。

## 命令行参考

| 命令或参数 | 默认值 | 含义 |
|---|---|---|
| `oracle-doa download` | slt:200，clb:60，其余五人各 40 | 抓取 CMU ARCTIC |
| `oracle-doa rirs --array` | `all` | 生成或检查 RIR 缓存 |
| `oracle-doa rirs --verify` | 关闭 | 打印各条件的 DRR |
| `oracle-doa arrays [--json]` | 表格 | 打印几何表 |
| `oracle-doa run --array` | `a2` | 阵列代号或 `all` |
| `--mask` | `irm,psm` | `irm`、`psm`、两者或 `none` |
| `--backend` | `all` | 五个名称的逗号列表 |
| `--utt` | 200 | 每条件句数 |
| `--snr` | -6 | babble 信噪比（dB），可给逗号列表 |
| `--t60`、`--dist` | 场景声明的网格 | 取网格子集 |
| `--seed` | 7 | 整轮共用一个随机数发生器 |
| `--rir-order` | 60 | 镜像源反射阶数 |
| `--data` | `$ORACLE_DOA_DATA` 或 `./data` | 语料与 RIR 缓存位置 |
| `--out` | `results/<array>_oracle.json` | 输出路径 |

`--t60` 与 `--dist` 必须落在声明的网格上。RIR 缓存按整个网格一次建好，网格外的取值没有对应的脉冲响应可读。

一轮评测共用一个随机数流，结果与 `--utt`、条件顺序绑定。比较不同设置时保持 `--utt` 与 `--seed` 一致，或只看聚合统计量。

## 输出

`results/<array>_oracle.json` 含阵列描述、实验设置、逐条件结果与全条件汇总。每个指标块的键为 `"<掩码>/<后端>"`，字段如下：

| 字段 | 含义 |
|---|---|
| `acc_tol` | 落在该阵列自身容差内的百分比 |
| `acc_5`、`acc_7p5`、`acc_10` | 在固定容差下的百分比，可跨阵列横向比较 |
| `mae_deg`、`rmse_deg` | 角度误差绝对值的圆周均值与均方根 |
| `n` | 该数字背后的句数 |

`examples/results/` 放了一次短跑与一次 a2 全量跑的结果，供对照。

### 示例：阵列 `a2`，每条件 200 句

下表为 5° 容差内的准确率（%）。DRR 取第一个麦克风的直达混响比；T60 = 0 时没有混响，DRR 在形式上无界。

| 系统 | 0.00 s | 0.20 s | 0.40 s | 0.60 s | 0.80 s | 1.00 s | 均值 | MAE |
|---|---|---|---|---|---|---|---|---|
| DRR（dB） | +146.4 | +4.0 | -2.3 | -4.8 | -6.5 | -7.7 | | |
| `gcc_phat`，无掩码 | 54.5 | 37.5 | 37.5 | 16.0 | 13.0 | 14.5 | 28.8 | 32.86 |
| `music`，无掩码 | 74.0 | 63.5 | 45.0 | 16.0 | 13.5 | 16.0 | 38.0 | 30.32 |
| `mgcc`，IRM | 100.0 | 99.0 | 99.5 | 98.5 | 97.0 | 98.0 | 98.7 | 0.36 |
| `sr_snr`，IRM | 100.0 | 99.5 | 100.0 | 100.0 | 99.5 | 99.0 | 99.7 | 0.34 |
| `sv`，IRM | 100.0 | 98.5 | 99.5 | 97.5 | 95.0 | 97.5 | 98.0 | 0.45 |
| `mgcc`，PSM | 100.0 | 99.0 | 100.0 | 100.0 | 99.5 | 100.0 | 99.8 | 0.20 |
| `sr_snr`，PSM | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 | 100.0 | 99.9 | 0.23 |
| `sv`，PSM | 100.0 | 99.0 | 100.0 | 100.0 | 99.0 | 99.5 | 99.6 | 0.29 |

掩码行与无掩码行的差距就是上界的量级。加掩码的 `music` 与不加掩码的结果在五种阵列上完全相同：MUSIC 逐对计算，每对只有一个一维信号子空间，正的掩码权重只改变频带权重，不动逐频带的峰值位置。

## 数据

仓库只提交代码。`data/` 与所有音频、缓存、权重文件都在 `.gitignore` 内。

**目标语音与 babble 语音：CMU ARCTIC。** 可自由使用，16 kHz，多说话人。项目主页 <http://www.festvox.org/cmu_arctic/>。单句按下列地址抓取：

```
http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/arctic_{set}{i:04d}.wav
```

其中 `set` 取 `a` 或 `b`，`i` 从 1 开始。`oracle-doa download` 默认取 `slt` 200 句、六个 babble 说话人各 40 到 60 句；这个配额决定语料池大小，语料池决定单个条件能取到多少不重复的句子，`--utt` 不得超过池长。

预期目录结构：

```
data/
  arctic/slt/arctic_a0001.wav ...
  arctic/{bdl,rms,jmk,awb,ksp,clb}/arctic_a0001.wav ...
  rirs_a2_o60.npz           # 由 `oracle-doa rirs` 生成
```

**本仓库不分发 IEEE 语料。** 那套 720 句录音（IEEE Recommended Practice for Speech Quality Measurements，IEEE 297-1969）没有公开下载渠道，需通过 IEEE 或授权渠道获取。本链路不绑定特定语料，任何 16 kHz 单人语料都能替换。下面两套同样使用 IEEE 句表、许可开放：

- ARU Speech Corpus（利物浦大学）：12 位说话人 × 720 句 IEEE 句子，无回声室录制，CC-BY。<https://datacat.liverpool.ac.uk/681/1/ARU_Speech_Corpus_v1_0.zip>
- Open Speech Repository：Harvard 句表，16 kHz 与 8 kHz 两个版本，多位说话人。<https://www.voiptroubleshooter.com/open_speech/american.html>

替换语料时请重采样到 16 kHz，放到 `data/arctic/<说话人>/` 下；或用 `--data` 指向别处的语料目录。换语料会改变无缠绕带宽，数字随之变化。

## 开销

| 步骤 | 规模 | 说明 |
|---|---|---|
| RIR 缓存，`a2` | 876 条脉冲响应 | 6 个 T60 × 73 个方向 × 2 类 |
| RIR 缓存，`b8`、`spk6`、`rob5` | 各 156 条 | 2 个距离 × 3 个 T60 × 13 个方向 × 2 类 |
| RIR 缓存，`car4` | 78 条 | 1 个距离 × 3 个 T60 × 13 个方向 × 2 类 |
| babble bank，`a2` | 438 条流，常驻约 340 MB | 1 个距离 × 6 个 T60 × 73 个方向，其中 37 个方向带 talker |
| babble bank，其余阵列 | 78 条以内 | 随网格规模等比缩小 |
| a2 全量跑，200 句 | 单核 7 分 33 秒 | 6 个条件，含 babble bank 构建 |

RIR 缓存写一次、之后只读。内存较小的机器建议按阵列分批生成，不要一次全部建完。

## 目录与测试

```
src/oracle_doa/
  config.py      常量与数据目录解析
  arrays.py      五种几何与方向网格
  rooms.py       Sabine 吸收、镜像源 RIR、磁盘缓存、DRR
  audio.py       语料池、STFT、babble bank、混音合成
  masks.py       oracle IRM 与 PSM
  backends.py    打分核与五个估计器
  evaluate.py    条件网格、评测循环、指标、JSON
  download.py    CMU ARCTIC 抓取
  cli.py         命令行
tests/           几何、掩码、合成输入上的后端、冒烟
examples/results/  一次短跑与一次 a2 全量跑
```

```bash
pytest -q
```

语料缺失，或本机无法用 `pyroomacoustics` 计算脉冲响应时，冒烟测试会自动跳过。

## 引用

掩码引导定位的框架与掩码定义沿用

> Z.-Q. Wang, X. Zhang, and D. Wang, "Robust Speaker Localization Guided by
> Deep Learning-Based Time-Frequency Masking," *IEEE/ACM Trans. Audio, Speech,
> Lang. Process.*, vol. 27, no. 1, pp. 178-188, 2019.

把掩码从阵列改由单通道估计，这条思路来自两阶段 CASA：

> Y. Liu and D. L. Wang, "Divide and Conquer: A Deep CASA Approach to
> Talker-Independent Monaural Speaker Separation," *IEEE/ACM Trans. Audio,
> Speech, Lang. Process.*, vol. 27, no. 12, pp. 2092-2102, 2019.

## 许可

本仓库代码采用 MIT 许可。各语料沿用各自的许可，本仓库不再分发其中任何一份。
