##### 调试环境

14服务器压力测试

```bash
nvidia-docker run -itd \
    -v /disk5/changchun/object_detection_stress/object_detection:/object_detection \
    -v /disk5/changchun/object_detection_stress/data:/object_detection/data \
    -v /disk5/changchun/object_detection_stress/config:/object_detection/config \
    -v /disk5/changchun/object_detection_stress/video:/object_detection/video \
    -v /disk5/changchun/object_detection_stress/logs:/object_detection/logs \
    -v /disk5/changchun/object_detection_stress/picture:/object_detection/picture \
    -v /disk5/changchun/object_detection_stress/model:/object_detection/model \
    -e CONFIG_PATH=/object_detection/config/config \
    -e SERVICE_NAME=object-detection-1 \
    --name detstr \
    --net=host \
    harbor.tianrang.com/traffic/cuda10.1_opencv:0.0.4 \
    /bin/bash
```

