# SailAlgo

My Algorihtm Library

- main.py: The Entry for Application
- LLM
 - prereq
   - transformers
   - sentence_transformers
   - trl
   - peft
 - use_m3e_embedding: use the huggingface m3e embedding model in data/pretrained/embedding/m3e_base 
- AIGC
  - prereq
    - transformers
    - huggingface_hub
    - hf_transfer
    - diffusers
  - diffusion.py 
  - gan.py 
  - normalizing_flow.py
  - vae.py 
  - vqvae.py 
- CG
  - ray_marcher_np