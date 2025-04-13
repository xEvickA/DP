  # ODHAD POLOHY OBJEKTOV SO 6 STUPŇAMI VOĽNOSTI S VYUŽITÍM RIEDKYCH 3D MODELOV RUKAMI DRŽANÝCH OBJEKTOV

  Odhad 6D polohy spočíva v dvoch fázach - predpríprave dát a spustenie OnePose. Predpríprava dát je vykonávaná pomocou estimate_6Dpose.py scriptu. 

  ## Inštalácia

  Pre spustenie scriptu estimate_6Dpose.py je potrebné si nainštalovať HOISTFormer podľa návodu v repozitári [HOISTFormer](https://github.com/xEvickA/HOISTFormer) do rovnakého foldera ako je script. 

  ## Spustenie

  Script sa spúšťa cez príkazový riadok `python estimate_6Dpose.py --video_path <cesta k priecinku s videom> --fps <fps>`, pričom fps je nepovinný argument.
  Tento script pripraví vstupné dáta pre OnePose, ktorý je treba nainštalovať podľa návodu v repozitári [OnePose](https://github.com/xEvickA/OnePose).
