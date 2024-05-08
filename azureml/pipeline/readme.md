To creat aml pipelines:
az ml job create --file <pipeline.yml>
```bash
az ml job create --file pipeline/pipeline_train.yml
az ml job create --file pipeline/pipeline_batch_score.yml
az ml job create --file pipeline/pipeline_prepare_data.yml

```