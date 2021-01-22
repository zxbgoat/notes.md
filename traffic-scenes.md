```bash
docker run -d \
           -v /home/tesla/demot/traffic-scenes/output:/traffic-scenes/output \
           -v /home/tesla/demot/traffic-scenes/config:/traffic-scenes/config \
           -v /home/tesla/demot/traffic-scenes/mask:/traffic-scenes/mask \
           harbor.tianrang.com/traffic/traffic-scenes:0.7 \
           config/nonmotor.yaml
```

```bash
docker run -itd \
           -v /disk5/changchun/traffic-scenes/nonmotor_obstruct/output:/traffic-scenes/output \
           -v /disk5/changchun/traffic-scenes/nonmotor_obstruct/config:/traffic-scenes/config \
           -v /disk5/changchun/traffic-scenes/nonmotor_obstruct/mask:/traffic-scenes/mask \
           --name tsno \
           harbor.tianrang.com/traffic/traffic-scenes:0.7 \
           config/nonmotor_obstruct.yaml
```

