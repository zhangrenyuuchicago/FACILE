for epoch in `seq 2 2 100`
do
    python save_feat.py --model_file save/SupCEWSI/TCGA_models/SupCEWSI_TCGA_dinov2_vitb14_lr_0.0125_decay_0.0001_bsz_64_trial_0_cosine/ckpt_epoch_$epoch.pth
    #sh run_test.sh > save/SupCEWSI/log_5way_epoch$epoch
done

