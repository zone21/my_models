#!/usr/bin/env bash
export LANG=zh_CN.UTF-8
##export HOROVOD_TIMELINE=/home/centos/autoimage2/denseNet_car/log/timeline.json

RUN_PATH="/home/centos/autoimage2/denseNet_car"
PRG_KEY="mpirun"
np=8
hostlist=192.168.13.211:2,192.168.12.239:2,192.168.13.212:2,192.168.12.235:2

cd $RUN_PATH

case "$1" in
    start)
        pid=$(pgrep -f $PRG_KEY)
        if [[ $pid -gt 0 ]]; then
            echo ""
            exit 1
        fi

        nohup mpirun -np $np \
          -H $hostlist\
          -mca pml ob1 \
          -mca btl ^openib \
          -bind-to none \
          -map-by slot \
          -x NCCL_DEBUG=INFO \
          -x LD_LIBRARY_PATH \
          -x NCCL_IB_CUDA_SUPPORT=1 \
          -x PATH \
          python3 $RUN_PATH/densenet_car_distr.py 2>&1 &

        echo "$PRG_KEY started, please check log."

        ;;

    stop)
        pid=$(pgrep -f $PRG_KEY)
        if [[ $pid -gt 0 ]]; then
            kill -9 $pid
            echo "$PRG_KEY stoped!"
        else
            echo "$PRG_KEY not found, nothing to stop!"
        fi

        ;;

    restart)
        $0 stop
        sleep 1
        $0 start

        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit 0