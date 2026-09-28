set -e
timestamp=$(date +%m%d%y_%H%M)

n_jobs=1
cv_split_n_list='0 1 2'
n_epochs=500
batch_size=64
donor_col='eid'
celltype_ABS='none'
mixmil_ref='main'
balancing_list='balanced'
target_col_list='AB1_INTESTINAL_INFECTIONS ATOPIC_DERM C3_OTHER_SKIN_EXALLC C3_SKIN_EXALLC E4_LIPOPROT H8_EXTERNAL H8_OTHEREAR I9_ANGINA I9_VARICVESOTH J10_ASTHMA_EXMORE J10_PNEUMONIA K11_CHOLELITH K11_DIAHER K11_DIVERTIC'
# CD2_BENIGN_OTHERDIGESTIVE K11_CONSTIPATION M13_FIBROBLASTIC M13_LIMBPAIN J10_ACUTELOWERNAS M13_SPONDYLOSIS K11_HERING K11_HERNIA K11_OTHDISSTOMDUOD K11_REFLUX M13_DORSALGIA
embedding_name_list='scanvi_emb'
seed=42 

for target_col in $target_col_list; do
    for embedding_name in $embedding_name_list; do
        for balancing in $balancing_list; do
            for cv_split_n in $cv_split_n_list; do
                outdir=MixMIL/results/disease_f3_v3/${timestamp}_mixmil_gpu__new_f3_v3___nepochs_${n_epochs}__${target_col}_cvsplit${cv_split_n}_${balancing}_embed_${embedding_name}_seed${seed}
                codedir=Giuditta/${timestamp}_gpu_codedir__new_f3_v3__${target_col}_cvsplit${cv_split_n}_${balancing}_embed_${embedding_name}_seed${seed}
                base_path=MixMIL/data/
                cv_table_path=cv_table_20251001__f3__only_european.csv
                adata_path=Freeze3-SCANVI-UMAP-metadata-obs--fullQCedCells-UKB.h5ad
                disease_table_path=cases_controls_all_category_nocancer.tsv

                dx mkdir -p $outdir
                dx mkdir -p $codedir
                dx rm -r $codedir
                dx mkdir -p $codedir

                dx upload run.py --path $codedir/run.py
                dx upload run.sh --path $codedir/run.sh
                dx upload personal_functions_torch.py --path $codedir/personal_functions_torch.py


                for ((job_id = 0; job_id < n_jobs; job_id++)); do
                  dx run --brief --destination $outdir swiss-army-knife \
                    -iin=$codedir/run.py \
                    -iin=$codedir/run.sh \
                    -iin=$codedir/personal_functions_torch.py \
                    -iin="data/freeze3/${adata_path}" \
                    -iin="${base_path}${cv_table_path}" \
                    -iin="Daniela/data_F3/disease/${disease_table_path}" \
                    --instance-type mem2_ssd1_gpu_x16 \
                    --priority high \
                    --name "mixmil_f3_new_v3_${job_id}_${target_col}_nepochs${n_epochs}_cvsplit${cv_split_n}_${balancing}_${embedding_name}" \
                    -iimage=nvcr.io/nvidia/pytorch:25.03-py3 \
                    -icmd="MIXMIL_REF=$mixmil_ref bash run.sh --job_id ${job_id} --n_jobs $n_jobs --cv_split_n $cv_split_n --n_epochs $n_epochs --batch_size $batch_size --celltype_ABS $celltype_ABS --balancing $balancing --target_col $target_col --embedding_name $embedding_name --donor_col $donor_col --cv_table_path $cv_table_path --adata_path $adata_path --seed $seed --disease_table_path $disease_table_path" \
                    --yes
                done
            done
        done
    done
done

#target_col_list = I9_IHD K11_IBS K11_DIVERTIC AB1_INTESTINAL_INFECTIONS E4_HYTHY_AI_STRICT JOINTPAIN D3_ANAEMIANAS D3_ANAEMIA_IRONDEF E4_HYPERCHOL G6_MIGRAINE E4_LIPOPROT J10_PNEUMONIA J10_ASTHMA_EXMORE I9_MI_STRICT ATOPIC_DERM T2D I9_ANGINA J10_CHRONSINUSITIS I9_HYPTENSESS L12_SEBORRKERAT

#dx rm -r $codedir ##needed to remove this row, otherwise it failed
#--instance-type mem2_ssd1_gpu_x16,  # also tried : mem2_ssd2_gpu1_x4, but oom error
# if mem2_ssd1_gpu_x16 goes in oom -> try mem2_ssd2_gpu1_x8 -> try mem2_ssd2_gpu1_x16 -> ...
# already run:
#target_col_list='  J10_ASTHMA_EXMORE T2D'
#['', 'C3_BREAST_EXALLC', 'C3_OTHER_SKIN_EXALLC',
 #      'C3_SKIN_EXALLC', 'CARDIAC_ARRHYTM', 'CD2_BENIGN_COLORECANI',
  #     'CD2_BENIGN_LIPOMATOUS', 'CD2_BENIGN_OTHERDIGESTIVE', 'D3_ANAEMIANAS',
   #    '', 
    #   ' 'G6_MIGRAINE', 'H7_CATARACTSENILE',
     #  'H7_CONJUNCTIVITIS', 'H8_EXTERNAL', 'H8_EXTOTITIS', 'H8_OTHEREAR',
      # 'I9_AF', '', 'I9_CORATHER', 'I9_HYPTENSESS', 'I9_HYPTENS',
      # ', '', 'I9_VARICVESOTH', 'I9_VARICVE',
      # 'J10_ACUTELOWERNAS', '', '',
      # 'J10_PERITONSABSC', 'J10_PHARYNGITIS', '', 'J10_SINUSITIS',
      # '', 'K11_CHOLELITH', 'K11_CONSTIPATION', 'K11_DIAHER',
      # '', 'K11_HERING', 'K11_HERNIA', '',
      # 'K11_OTHDISSTOMDUOD', 'K11_REFLUX', 'L12_FOLLICULARCYST',
      # 'L12_SEBORRKERAT', 'M13_CERVICALGIA', 'M13_DORSALGIA',
      # 'M13_FIBROBLASTIC', 'M13_HALLUXVALGUS', 'M13_LIMBPAIN', 'M13_SCIATICA',
      # 'M13_SPONDYLOSIS', 'N14_GLOMEINOTH', 'SLEEP', '',
      # 'Z21_KIDNEY_TRANSPLANT_STATUS', '', 'MDD'

      # todo december: 
      #'C3_BREAST_EXALLC E4_OBESITY I9_IHD E4_HYTHY_AI_STRICT E4_HYPERCHOL T2D CARDIAC_ARRHYTM CD2_BENIGN_COLORECANI D3_ANAEMIA_IRONDEF I9_AF I9_CORATHER I9_HYPTENS I9_MI_STRICT L12_SEBORRKERAT MDD'
