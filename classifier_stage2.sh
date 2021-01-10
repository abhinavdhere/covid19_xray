#!/bin/bash
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 1 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 1e-4 -lwts 0.3,0.3,0.4 --foldNum 0 --amp True --gamma 3
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 11 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 5e-5 -lwts 0.3,0.3,0.4 --foldNum 0 --amp True --gamma 3 -loadflg main
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 21 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 5e-5 -lwts 0.4,0.3,0.3 --foldNum 0 --amp True --gamma 3 -loadflg main
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 31 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 5e-5 -lwts 0.4,0.3,0.3 --foldNum 0 --amp True --gamma 2 -loadflg main
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 41 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 5e-5 -lwts 0.4,0.3,0.3 --foldNum 0 --amp True --gamma 2 -loadflg main
echo "Testing with chkpt"
python3 -u learner.py --saveName covidx_stage2_wSeg_FL_pairAug_attn_fold0  --initEpochNum 41 --nEpochs 10 --batchSize 64 -wd 1e-4 -lr 5e-5 -lwts 0.4,0.3,0.3 --foldNum 0 --amp True --gamma 2 -loadflg chkpt --runMode val
