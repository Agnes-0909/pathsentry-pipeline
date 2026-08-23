#!/bin/bash
# 停止采集：优雅结束（完成队列落盘并打印统计），并列出本次产物目录
PIDFILE=/root/capture_latest.pid
[ -f $PIDFILE ] || { echo "没有正在进行的采集"; exit 0; }
kill -TERM $(cat $PIDFILE) 2>/dev/null
for i in $(seq 1 30); do kill -0 $(cat $PIDFILE) 2>/dev/null || break; sleep 1; done
kill -0 $(cat $PIDFILE) 2>/dev/null && kill -KILL $(cat $PIDFILE)
rm -f $PIDFILE
echo "--- 本次统计 ---"; tail -2 /root/capture_logs/latest.log
echo "--- 最新产物目录 ---"; ls -dt /root/data/raw/*/ | head -1
