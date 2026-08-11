# Obstacle 插件：室内(png)/室外(jpg) 障碍物距离检测

在 main 基础上新增的障碍物感知接口，部署在 Jetson（TensorRT 10.4.0）。

## 接口（MCP tool `obstacle`）

JSON-RPC POST `http://<host>:15720/mcp`，`tools/call`：

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"obstacle","arguments":{"action":"info"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call",
 "params":{"name":"obstacle","arguments":{"action":"detect","image_path":"/data/frame.png"}}}
```

| 参数 | 说明 |
|---|---|
| action=info | 插件/引擎状态 |
| action=detect | 障碍物检测；`image_path` 必填 |
| mode | auto（默认）/ indoor / outdoor；auto 按扩展名：**png=室内、jpg/jpeg=室外** |

返回：
```json
{"ok": true, "mode": "outdoor", "distance_m": 4.123, "fallback": false,
 "image_path": "/data/frame.jpg", "elapsed_ms": 38.2}
```

## 两条管线

| 模式 | 引擎 | 后处理 | 输出 |
|---|---|---|---|
| 室内 (png) | DA2-Small metric-hypersim INT8（[1,3,518,686]） | ROI(0-300,213-426) min → isotonic 标定 → clip[0.05,50] | 距离(m) |
| 室外 (jpg) | yolo26n-depth INT8(768) + yolo26n-seg FP16(640) | 掩码(allowed 类,conf≥0.25) → p5 of max(depth-1,0) → 1.15·d−1.5 → clip[0,80] | 距离(m) |

室外引擎可换 `yolo26n-seg_int8.trt`（config 改 `seg_engine`）；当前默认 depth INT8 + seg FP16（INT8 seg 的掩码精度略低，F1@5m 降 ~0.08）。

## 仅运行 obstacle

obstacle 部署只加载障碍物接口，其它服务全部关闭：
- 插件：asr / tts / htmsg / vop 均 `enabled: false`，仅 `obstacle.enabled: true`
- WebSocket ASR 服务器：main.py 按 `plugins.asr.enabled` 启动，关闭时不再监听 ws_port
- agent-core 注册心跳：`register: false` 时关闭

Jetson 实测（2026-08-09，端口 15730）：日志仅出现 `ObstaclePlugin loaded` /
`MCP server`，无 ws_asr / registration 线程；室内 png → 2.468m、室外 jpg(GT=3.89m) → 4.012m。

## 配置

- `perception/config.yaml`：`plugins.obstacle.enabled: false`（默认关，不影响既有镜像）
- `perception/config.obstacle.yaml`：obstacle 启用版（obstacle 镜像用）
- 模型目录 `/models/obstacle/`：室内 int8 引擎+isotonic json；室外 depth_int8 / seg_fp16 引擎

## Jetson 实测（2026-08-09）

镜像 `dustynv/ros:obstacle-trt`（TRT 10.4.0）+ 挂载代码/模型，MCP 端口 15730：

| 测试 | 输入 | 返回 | 说明 |
|---|---|---|---|
| tools/list | - | 出现 `obstacle` 工具 | 注册正常 |
| obstacle/info | - | state=ready | 三引擎+标定加载成功 |
| obstacle/detect | indoor.png（TUM） | distance_m=2.468，152ms | 室内管线 |
| obstacle/detect | outdoor_near.jpg（nuScenes，GT=3.89m 车） | distance_m=4.012，231ms | 室外管线，误差 0.12m |

**版本坑**：室内引擎原先在 TRT 10.3 宿主构建，TRT 10.4 运行时反序列化报
`engine plan file is not compatible`；已在 10.4 镜像内重建（int8 52.2MB / fp16 54.1MB），
并替换 JuiceFS `obstacle_val_trt/` 下载源。**obstacle 部署链路里所有引擎必须是 TRT 10.4 构建。**

## 构建（Jetson 上）

```bash
cd phanthymotus-submit
docker build -f perception/Dockerfile.obstacle --network=host -t phanthymotus-obstacle:latest .
docker run -d --runtime=nvidia --network=host \
  --privileged -v /dev:/dev -v /opt/embodied/models:/models \
  phanthymotus-obstacle:latest
```

Dockerfile 内已注入修复 TRT python 所需的 Jetson DLA/驱动库（镜像原为 0 字节空文件），并自动下载 4 个引擎/标定文件。

## 验证

- `curl -X POST http://localhost:15720/mcp -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'` → 应出现 `obstacle`
- `obstacle/info` → state=ready
- 用室内 png / 室外 jpg 各测一张 `obstacle/detect`

## ROS2 接口（基准线）

`ObstacleDistancePlugin` 支持 ROS2 订阅推理（vop 同款多实例节点）：

- `action=start`，参数 `input_topic`（如 `/benchmark/camera/image/obstacle_local_10`）→ 创建节点
  - 订阅 `sensor_msgs/CompressedImage`（支持 png/jpg，大图需 FastDDS 大消息配置 `perception/config/fastdds_large_message.xml`）
  - 按 `format` 分流：png → 室内，jpg/jpeg → 室外
  - 发布 `std_msgs/String` JSON `{"pred_distance": <float>}` 到 `{input_topic}/obstacle_distance`（可用 `output_topic` 配置覆盖）
- `action=stop` / `action=config` / `action=info` 管理节点生命周期
- MCP `action=detect`（image_path）与 ROS2 路径并存

示例发布：
```
[obstacle] ros2 result: topic=.../obstacle_distance mode=indoor pred_distance=2.468 fallback=False n=1
```

## 内存优化（8GB Jetson）

- **缓冲复用**：`_TrtEngine` 缓存 device/pinned-host 缓冲，跨帧复用，不再每帧 cudaMalloc/Free（消除分配抖动与碎片）。
- **按模式懒加载**：`lazy_load: true`（默认开）——室内/室外引擎按需加载，只跑室内就不加载室外引擎，省显存；`info` 返回 `loaded` 列表。首帧会多 1-2s 加载时间。
- 运行期系统内存基线（Jetson 8GB）：本服务约数百 MB，与 TTS/agent-core 共存时注意剩余内存；benchmark 前可停掉不需要的容器。

## 室内 2m 推边（F1@2m 0.64 → 0.76）

室内管线在 isotonic 回归后加 2m 准召线推边（zeng 同款机制）：
- `d_roi_min < push_score_threshold_m(1.86)` → 近侧：pred = min(pred, decision_threshold_m - margin)
- 否则 → 远侧：pred = max(pred, decision_threshold_m)
- 标定 json 恢复为无 bias 全量 isotonic（推边接管边界，bias 只伤 MAE）

TUM 2200 帧 5 折 CV：F1@2m=0.764±0.008（P=0.759 R=0.769），MAE=0.347；全量 0.764/MAE 0.31。

**残差校正（v2）**：isotonic 输出在 pred 空间存在 +0.14~0.31m 局部上偏（PAVA 线性插值所致），
在标定 json 中加入 `rx_knots/ry_knots` 残差曲线：`base -= interp(base, rx, ry)`，再推边。
组合后：F1@2m=0.764（不变），MAE 0.347→0.328（5 折）/0.284（全量）。
