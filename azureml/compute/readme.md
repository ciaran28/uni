To creat aml compute:
az ml compute create -- file <file_name.yml>
```bash
az ml compute create --file compute/cpu-cluster-D13.yml

az ml compute create -- file compute/gpu-V100.yml

az ml compute create --file compute/cpu-cluster.yml
```