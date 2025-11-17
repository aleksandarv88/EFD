======================================================
sims/README.txt
======================================================

This folder contains simulations for the {chapter_name} chapter.

Each simulation lives in:

  {sim_name}/

and follows this structure:

  {sim_name}/
    README.txt        ← what this sim studies

    gpt/
      README.txt
      {sim_name}_CODE_gpt.py
      {sim_name}_RESULT_gpt.txt
      config/
        *.yaml
      media/
        images/
        videos/
        data/

    grok/
      ...

    gemini/
      ...

To explore a simulation:

  1. Read {sim_name}/README.txt
  2. If needed, inspect {ai_name}/README.txt for a specific AI run.
  3. Look at media/ for plots, images, metrics.

======================================================
END OF sims/README
======================================================
