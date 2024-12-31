To start ray on the master
```
ray start --head --port=6379 --num-cpus=15 --node-ip-address='192.168.1.221'
```

To start ray on the nodes
```
PYTHONPATH="$PYTHONPATH:$PWD" RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address='192.168.1.221:6379'
```