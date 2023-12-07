for epoch in `seq 1 1 20`
do
    python save_feat.py --model_file save/SupCETile/TCGA_models/SupCETile_TCGA_dinov2_vitb14_lr_0.05_decay_0.0001_bsz_64_trial_0_cosine/ckpt_epoch_$epoch.pth 
    sh run_test.sh > save/SupCETile/log_NCT_9way_5shot_epoch$epoch
done
