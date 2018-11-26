##export HOROVOD_TIMELINE=/home/centos/autoimage2/denseNet_car/log/timeline.json

nohup mpirun -np 2 \
  -H 192.168.13.211:2\
  -mca pml ob1 \
  -mca btl ^openib \
  -bind-to none \
  -map-by slot \
  -x NCCL_DEBUG=INFO \
  -x LD_LIBRARY_PATH \
  -x NCCL_IB_CUDA_SUPPORT=1 \
  -x PATH \
  python3 /home/centos/autoimage2/denseNet_car/densenet_car_distr.py 2>&1 &
