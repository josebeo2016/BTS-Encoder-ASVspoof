Training: 
```
./train.sh <batch size>
```

Config: model_config_RawNet_first.yaml for big RNN, transformer
config: model_config_RawNet for small CNN (embedding size is smaller)


pretrained models in: models_1st

There are different models, need to modify code in model.py before training for evaluating

```
            # x = x + bio_scoring[:,:,-1] # add the conditioning bio scoring CNN
            # x = x + bio_scoring # add the conditioning bio scoring RNN
            x = torch.cat((x, bio_scoring), 1)
```

and

```
        # self.bioScoring = bioEncoderConv(d_args, self.device)
        # self.bioScoring = bioEncoderRNN(d_args, self.device)
        # self.bioScoring = bioEncoderTransformer(d_args, device)
        # self.bioScoring = bioEncoderlight(d_args, self.device)
        self.bioScoring = bioEncoderRNNsmall(d_args, self.device)
```