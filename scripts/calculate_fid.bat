@echo off

python ".\8 - Calculate FID.py" ^
  --real_dir ..\datasets\flat\fold_1 ^
  --fake_dir ..\datasets\flat\fake_batch-16_lr_1e-5_fold_1_128px ^
  --output ..\data\result_batch-16_lr_1e-5_fold_1_128px

python ".\8 - Calculate FID.py" ^
  --real_dir ..\datasets\flat\fold_3 ^
  --fake_dir ..\datasets\flat\fake_batch-16_lr_1e-5_fold_3_128px ^
  --output ..\data\result_batch-16_lr_1e-5_fold_3_128px

python ".\8 - Calculate FID.py" ^
  --real_dir ..\datasets\flat\fold_1 ^
  --fake_dir ..\datasets\flat\fake_batch-16_lr_5e-6_fold_1_128px ^
  --output ..\data\result_batch-16_lr_5e-6_fold_1_128px

python ".\8 - Calculate FID.py" ^
  --real_dir ..\datasets\flat\fold_3 ^
  --fake_dir ..\datasets\flat\fake_batch-16_lr_5e-6_fold_3_128px ^
  --output ..\data\result_batch-16_lr_5e-6_fold_3_128px

