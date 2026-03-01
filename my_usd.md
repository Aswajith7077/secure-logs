

You can change the config for 2K-dataset, 100K-dataset, and Full-dataset by changing the config/config.py file.
```python
config_service = ConfigService(ConfigService.CATEGORIES[0]) # 2K-dataset
config_service = ConfigService(ConfigService.CATEGORIES[1]) # 100K-dataset
config_service = ConfigService(ConfigService.CATEGORIES[2]) # Full-dataset
```

All these configs are defined in the .env files in the config folder.
Also automatically the device will be set to cuda if available or cpu otherwise.

Training:

```bash
python main.py
```

For Inference and predictions:

This one generates all the visualizations and evaluation metrics under `result/2k` or `result/100k` or `result/full` depending on the config.

Also, the models cannot be uploaded to the hugging face hub because it requires an access token.


```bash
python inference/predict.py
```

The Sessions will be of the form

```python
{
    'blk_4343207286455274569': {
        'has_anomaly': False,
        'templates': [
            'Receiving block blk_<*> src: '
            '/<*>:<*> dest: /<*>:<*>'
        ]
    },
}
```

And Templates

```python
{
    'blk_4343207286455274569': [
        'Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>'
    ]
}
```