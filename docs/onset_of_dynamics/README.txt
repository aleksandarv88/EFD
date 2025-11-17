======================================================
docs/onset_of_dynamics/README.txt
======================================================

CHAPTER: ONSET OF DYNAMICS

This chapter connects the abstract substrate (Ω) and
emergence logic to proto-physics: geometry, curvature,
and continuum behavior.

Focus:
  - how discrete loop structures approximate smooth surfaces
  - how curvature and metrics emerge from angle defects
  - how refinement and convergence behave
  - how this links back to EFD foundations

======================================================
DOCUMENTS
======================================================

1. EFD_ONSET_OVERVIEW_1.0.txt   (if present)
   - High-level conceptual intro:
       - what "onset of dynamics" means in EFD
       - connection to Ω, M, ∂M, η, stable loops
       - what the simulations demonstrate

If this file is missing, refer back to:

  docs/foundations/EFD_HIERARCHY_1.0.txt

for the conceptual context and then explore the sims.

======================================================
SIMULATIONS
======================================================

Simulations for this chapter live under:

  docs/onset_of_dynamics/sims/

and follow this structure:

  sims/
    README.txt          ← general info for all sims in this chapter

    {sim_name}/
      README.txt        ← what this particular sim does

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
        README.txt
        {sim_name}_CODE_grok.py
        {sim_name}_RESULT_grok.txt
        config/
          *.yaml
        media/
          ...

      gemini/
        README.txt
        {sim_name}_CODE_gemini.py
        {sim_name}_RESULT_gemini.txt
        config/
          *.yaml
        media/
          ...

For a concrete run (e.g. curvature_angle_defect), go to:

  docs/onset_of_dynamics/sims/curvature_angle_defect/README.txt

======================================================
END OF onset_of_dynamics/README
======================================================
