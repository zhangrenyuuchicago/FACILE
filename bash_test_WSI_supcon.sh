for epoch in `seq 20 20 200`
do
    python save_feat.py --model_file save/SupConWSI/TCGA_models/SupConWSI_TCGA_dinov2_vitb14_lr_0.0125_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine/ckpt_epoch_$epoch.pth
    sh run_test.sh > save/SupConWSI/log_9way_5shot_epoch$epoch
done

