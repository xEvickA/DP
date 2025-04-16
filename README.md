  # ODHAD POLOHY OBJEKTOV SO 6 STUPŇAMI VOĽNOSTI S VYUŽITÍM RIEDKYCH 3D MODELOV RUKAMI DRŽANÝCH OBJEKTOV

  Odhad 6D polohy je vykonávaný pomocou scriptu `run_estimation.sh`, ktorý najprv spustí `data_preprocessing.py` a následne spustí odhad 6D polohy pomocou modelu OnePose. 

  ![demo_video](assets/6d_video.gif)

  ## Inštalácia

  Pre spustenie scriptu `run_estimation.sh` je potrebné si nainštalovať HOISTFormer podľa návodu v repozitári [HOISTFormer](https://github.com/xEvickA/HOISTFormer) a OnePose podľa návodu v repozitári [OnePose](https://github.com/xEvickA/OnePose) do rovnakého foldera ako je script. 

  ## Spustenie

  Script sa spúšťa cez príkazový riadok `./run_estimation.sh <cesta k .mp4 súboru> <fps>`, pričom fps je nepovinný argument.
  
