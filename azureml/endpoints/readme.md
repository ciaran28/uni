To creat batch end points in aml:
az ml batch-endpoint create -- file <file_name.yml> --name <end-point-name>

```bash

ENDPOINT_NAME="titanic-batch-endpoint"
az ml batch-endpoint create --file endpoints/batch-endpoint.yml  --name $ENDPOINT_NAME

```