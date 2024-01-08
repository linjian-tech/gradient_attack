

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Restruct from RT dataset
```
# Start training with: 
python main.py 
parameter setting:
 
para = {'dataPath': 'dataset/rtMatrix.txt',
            'outPath': 'result/',
            'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # evaluate matric
            'density': 0.001, # dataset density
            'dimension': 10, # dimenisionality of the latent vectors
            'eta': 0.001, # learning rate
            'lambda': 0.01, # regularization parameter
            'reconstruct_max_iter': 100, # the max iterations for reconstruction
            'local_iter': 5,
            'thresholds': 0.1, # if loss < thresholds, then the reconstruction stops
            'continue_round': 3, # how many continue rounds gradients are collected
            'repeat_experiment':10, # how many runs are performed at each matrix density
            }
```

## Restruct from TP dataset
```
# Start training with: 
python main.py 
parameter setting:
 
para = {'dataPath': 'dataset/tpMatrix.txt',
            'outPath': 'result/',
            'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # evaluate matric
            'density': 0.001, # dataset density
            'dimension': 10, # dimenisionality of the latent vectors
            'eta': 0.0001, # learning rate
            'lambda': 0.01, # regularization parameter
            'reconstruct_max_iter': 100, # the max iterations for reconstruction
            'local_iter': 5,
            'thresholds': 0.5, # if loss < thresholds, then the reconstruction stops
            'continue_round': 3, # how many continue rounds gradients are collected
            'repeat_experiment':10, # how many runs are performed at each matrix density
            }

```


