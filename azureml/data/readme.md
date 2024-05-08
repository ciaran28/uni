To creat aml data assets:
az ml data create -- file <file_name.yml>
```bash
az ml compute create --file data/local-file.yml

az ml compute create -- file data/local-folder.yml

az ml compute create --file data/local-mltable.yml
```