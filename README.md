  # ODHAD POLOHY OBJEKTOV SO 6 STUPŇAMI VOĽNOSTI S VYUŽITÍM RIEDKYCH 3D MODELOV RUKAMI DRŽANÝCH OBJEKTOV

  Odhad 6D polohy je vykonávaný pomocou scriptu `run_estimation.sh`, ktorý najprv spustí `data_preprocessing.py` a následne spustí odhad 6D polohy pomocou modelu OnePose. 

  ![demo_video](assets/6d_video.gif)

  ## Inštalácia

  Pre spustenie scriptu `run_estimation.sh` je potrebné si nainštalovať HOISTFormer podľa návodu v repozitári [HOISTFormer](https://github.com/xEvickA/HOISTFormer) a OnePose podľa návodu v repozitári [OnePose](https://github.com/xEvickA/OnePose) do rovnakého foldera ako je script. 
  Po inštalácii by repozitár mal vyzerať nasledovne:
  ```
├── detectron2/
├── HOISTFormer/
├── Mask2Former/
├── OnePose/
├── data_preprocessing.py
├── poses_and_intrins.py
├── read_write_model.py
└── run_estimation.sh
   ```      
   
  ## Spustenie

  Script sa spúšťa cez príkazový riadok `./run_estimation.sh --video1=<cesta k .mp4 súboru, z ktorého je vytváraný 3D model> --video2=<cesta k .mp4 súboru, v ktorom je odhadovaný poloha objektu> --fps=<fps>`, pričom fps a video2 sú nepovinné argumenty, keď argument video2 nie je zadaný, odhad polohy sa vykoná na videu1. 
  
