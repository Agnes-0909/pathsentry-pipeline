#!/bin/bash
# PathSentry 双目采集启动（部署于 X5 板 /root/，默认配置全内置）
# 用法: ./start_capture.sh [场景标签]      如 ./start_capture.sh grass_backlight
# 默认: host 0/2、LPWM 硬同步、保存前顺时针旋转90°、JPEG q92、stride 3(约10fps)
# 输出: /root/data/raw/<场景>_<时间>/
SCENE=${1:-untitled}
systemctl start hobot-cam-service 2>/dev/null   # 相机供电依赖
mkdir -p /root/data/raw /root/capture_logs
LD_LIBRARY_PATH=/root/ps_libs nohup /root/ps_collector \
  -o /root/data/raw --scene "$SCENE" --stride 3 \
  > /root/capture_logs/latest.log 2>&1 &
echo $! > /root/capture_latest.pid
sleep 2
if kill -0 $(cat /root/capture_latest.pid) 2>/dev/null; then
  echo "采集中 pid=$(cat /root/capture_latest.pid) 日志=/root/capture_logs/latest.log"
  tail -1 /root/capture_logs/latest.log
else
  echo "启动失败:"; cat /root/capture_logs/latest.log
fi
