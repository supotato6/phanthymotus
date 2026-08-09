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

## 配置

- `perception/config.yaml`：`plugins.obstacle.enabled: false`（默认关，不影响既有镜像）
- `perception/config.obstacle.yaml`：obstacle 启用版（obstacle 镜像用）
- 模型目录 `/models/obstacle/`：室内 int8 引擎+isotonic json；室外 depth_int8 / seg_fp16 引擎

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
