# ps_collector 使用文档

板端双目数据采集工具的完整操作手册：编译 → 部署 → 采集 → 产物校验 → 故障排查。

---

## 1. 功能概览

| 能力 | 说明 |
|---|---|
| 双目同步采集 | 双路 MIPI SC132GS（1088×1280@30fps），LPWM 硬件触发，左右目同曝 |
| 帧配对 | 按硬件时间戳配对，容差默认 1ms；不齐自动丢旧帧重对齐 |
| 落盘 | 后台线程 NV12→BGR→JPEG（默认 q92）/PNG 无损；有界队列满则丢新帧，不拖慢采集 |
| 抽帧 | `--stride` 对 30fps 原始流抽帧（如 stride=3 → 有效 10fps） |
| 场景标签 | `--scene` 写入目录名与 meta.json，对接采集矩阵（草地/逆光/黄昏…） |
| 元数据 | index.csv（frame_id、硬件时间戳 ns、skew_ns）+ meta.json（会话配置） |
| 停止方式 | Ctrl-C / SIGTERM 优雅停止，或 `--max-pairs` / `--duration-sec` 自动停止 |

## 2. 编译

### 2.1 环境要求

- 宿主机/容器：Ubuntu 22.04（推荐使用 docker 容器 `RDK_x86`，内含工具链与 `/project/3rdlibrary`）
- 交叉工具链：`aarch64-linux-gnu-gcc / g++`、`cmake >= 3.14`
- 依赖库：`RDK_resource/RDK_proj/3rdlibrary/{RDK_CAMERA, opencv_aarch64}`

### 2.2 编译命令

```bash
cd 01_data/collect
./build.sh                      # 宿主机：默认 RDK_ROOT=../../../RDK_resource/RDK_proj

# 在 RDK_x86 容器内（依赖在 /project/3rdlibrary）：
docker cp 01_data/collect/. <容器>:/project/ps_collector/
docker exec <容器> bash -lc 'cd /project/ps_collector && RDK_ROOT=/project ./build.sh'
```

产物：`build_aarch64/ps_collector`（AArch64 ELF，PIE）。

### 2.3 常用编译变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RDK_ROOT` | `../../../RDK_resource/RDK_proj` | 3rdlibrary 所在工程根 |
| `BUILD_DIR` | `./build_aarch64` | 构建目录 |
| `BUILD_TYPE` | `Release` | CMake 构建类型 |

## 3. 部署到板子

```bash
scp build_aarch64/ps_collector root@<板子IP>:/root/
# 板端运行时依赖（若最小系统未带）：libvpf.so / libcam* / libhbmem.so，随系统镜像自带
```

## 4. 运行

### 4.1 典型命令

```bash
# 草地逆光场景，抽帧 1/3（约 10fps），限时 5 分钟
./ps_collector -o /root/data/raw --scene grass_backlight --stride 3 --duration-sec 300

# 室内场景，无损 PNG，采满 2000 对
./ps_collector -o /root/data/raw --scene indoor --format png --max-pairs 2000

# 长尾补采（电线/坑洞摆放后），全帧率手动控制启停
./ps_collector -o /root/data/raw --scene wire_closeup
```

### 4.2 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-o, --output DIR` | `data/raw` | 输出根目录 |
| `--scene TAG` | `untitled` | 场景标签（采集矩阵维度，见 docs/01_data.md §3.2） |
| `--format FMT` | `jpg` | `jpg` 或 `png` |
| `--jpg-quality N` | `92` | JPEG 质量 1-100 |
| `--left-host N` | `0` | 左目 MIPI host 编号 |
| `--right-host N` | `1` | 右目 MIPI host 编号 |
| `--timeout-ms N` | `1000` | 单帧获取超时 |
| `--max-pairs N` | `0`（不限） | 最多保存对数 |
| `--duration-sec N` | `0`（不限） | 最长采集时长（秒） |
| `--stride N` | `1` | 每 N 对保存一对 |
| `--max-skew-ns N` | `1000000` | 左右硬件时间戳配对容差（ns） |
| `-h, --help` | — | 帮助 |

### 4.3 停止与统计输出

Ctrl-C（SIGINT/SIGTERM）后程序完成队列中剩余帧的落盘再退出，并打印一行统计：

```
[ps_collector] 采集结束: acquired=4500 saved=1500 dropped=0 write_failures=0
sync_drops=37 timeouts=(0/0) max_skew=412000ns elapsed=300012ms
```

- `acquired/saved`：配对成功对数 / 实际保存对数（受 stride、max-pairs 影响）；
- `dropped`：落盘队列满丢掉的帧（应为 0，见 §6）；
- `sync_drops`：时间戳不齐丢弃重取的次数（少量正常，硬同步下一般 <1%）；
- `max_skew`：全程最大左右时间戳差（应 ≤ `--max-skew-ns`）。

## 5. 产物格式

```
<output>/<scene>_<YYYYmmdd_HHMMSS>/
├── left/000001.jpg ...      # 与 right 同序号严格一一配对
├── right/000001.jpg ...
├── index.csv
└── meta.json
```

**index.csv**（UTF-8，无空格）：

```csv
frame_index,left_frame_id,right_frame_id,left_ts_ns,right_ts_ns,skew_ns,left_file,right_file
0,128,128,1723690012345678901,1723690012345678501,400,/root/data/raw/.../left/000001.jpg,...
```

**meta.json**：会话配置快照（scene/format/stride/传感器型号/同步方式/host 编号等）。

回传宿主机：

```bash
scp -r root@<板子IP>:/root/data/raw/<scene>_* ./data/raw/
```

## 6. 采集质量验收（对应 docs/01_data.md §4）

1. **同步质量**：`awk -F, 'NR>1 {if ($6>1000000) n++} END {print n+0}' index.csv` 应为 0；
   统计行 `max_skew` ≤ 1e6 ns 是 LPWM 硬同步生效的直接证据。
2. **磁盘吞吐**：`dropped=0`；若非 0，加大 `--stride`、降低 `--jpg-quality`，或换更快存储。
3. **配对正确性**：抽查同序号 left/right 图像，运动物体位置应无可辨错位。
4. **采集前后标定**：每次采集前后对棋盘格各采 20 帧（`--scene calib --max-pairs 20`），
   双目重投影误差 <0.5px 才入库（docs/01_data.md §3.1）。

## 7. 故障排查

| 现象 | 原因与处置 |
|---|---|
| `SC132GS chip ID ... not found` | 相机未上电/接线错误/host 编号不对；核对 `--left-host/--right-host` 与 vcon 设备树 |
| `failed to read vcon rx_phy/lpwm_chn` | 设备树缺少 vcon 节点，检查镜像与相机树配置 |
| `too many consecutive frame timeouts` | ISP/VSE 起流失败或传感器掉线；重启程序，多次复现则查供电 |
| `hardware timestamp missing` | 时间戳通路异常（ts_src 路由错误）；不要降容差掩盖，先排查硬件 |
| `dropped` 持续增长 | 磁盘写不过来：`--stride` 调大 / 质量 降低 / 换 NVMe 或 eMMC 直写 |
| 保存图像有撕裂/绿边 | stride 处理异常，确认使用 `cvtColorTwoPlane` 路径未被改动 |

## 8. 二次开发指引

- 新增传感器：在 `src/capture/sensor/` 添加寄存器配置（参考 `mipi_real_time_proj/src/sensor/` 有 sc230ai/imx219 等现成配置），并在 `vio_dual.cpp` 的 `cloneSensorConfig` 中切换；
- 启用 IMU：经 `hbn_camera_parse_emb` 解析传感器嵌入式数据（规划项）；
- 模块边界：`capture` 只管取帧与配对，`storage` 只管编码落盘，二者经 `StereoFrame`（深拷贝内存）解耦——替换任一模块不影响另一个。
