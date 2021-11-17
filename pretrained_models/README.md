We provide full AudioSet and FSD50K pretrained models (click the mAP to download the model(s)).

|                                          | # Models  |AudioSet (Eval mAP) | FSD50K (Eval mAP) |
|------------------------------------------|:------:|:--------:|:------:|
| Single Model                           |  1 |[0.440](https://www.dropbox.com/s/d1z27wj30ew5qrs/as_mdl_0.pth?dl=1)  |  [0.559](https://www.dropbox.com/s/stzrmfty2oyqnnj/fsd_mdl_best_single.pth?dl=1) |
| Weight Averaging Model                 |  1 | [0.444](https://www.dropbox.com/s/ieggie0ara4x26d/as_mdl_0_wa.pth?dl=1)  |  [0.562](https://www.dropbox.com/s/5fvybrbulvhsish/fsd_mdl_wa.pth?dl=1) |
| Ensemble (Single Run, All Checkpoints) | 30/40 |[0.453](https://www.dropbox.com/sh/jo6te8fcy1ptabw/AAAtJ9sMn93-3L0XkebzQQxIa?dl=1)  |   [0.573](https://www.dropbox.com/sh/gyv95m53sib36vk/AADWCgApSxtAEVU1KrnQApi3a?dl=1)  |
| Ensemble (3 Run with Same Setting) | 3 |  [0.464](https://www.dropbox.com/sh/c83w8816vl6yhty/AADjoO9irfP1RCr-qyZMJg_-a?dl=1)  |   N/A  |
| Ensemble (All, Different Settings) | 10 |  [0.474](https://www.dropbox.com/sh/ihfbxcemxamihz9/AAD9zqnUptZzyZlquqpWllDya?dl=1)  |   N/A  |

All models are EfficientNet B2 model with 4-headed attention with 13.6M parameters, trained with 16kHz audio. Load the model by using follows:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
num_class = 527 if dataset=='audioset' else 200
audio_model = models.EffNetAttention(label_dim=num_class, b=2, pretrain=False, head_num=4)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)
```

We strongly recommend to use the pretrained model with our dataloader for inference to avoid the dataloading mismatch.

For ensemble experiments, uncompress the models and place in this folder:

```
pretrained_models
│   README.md
└───audioset
│   │   as_mdl_0.pth
│   │   as_mdl_1.pth
└───fsd50k
    │   fsd_mdl_0.pth
    │   fsd_mdl_1.pth
```