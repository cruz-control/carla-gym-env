This repository provides a CARLA Gymnasium environment, to be used by reinforcement learning for autonomous vehicle research experiments. 

Steps for setup:

prerequisites:
1. Install Carla 0.9.15: Check out the prerequisites and the installation guide for CARLA --> https://carla.readthedocs.io/en/latest/start_quickstart/#before-you-begin
2. Python 3.10: Required to match the CARLA 0.9.15 wheel.
3. NVIDIA GPU: requirement may vary depending on experiment however you can checkout the CARLA guide above for the minimum requirements.

installation:
1. Clone the repository
2. create and activate the conda environment (Python 3.10)
3. install dependencies 
    ```bash
   pip install gymnasium stable-baselines3 scipy scikit-image pillow tensorboard
   pip install torch torchvision
    ```
4. Modify run.py with the port number you are running CARLA on, generally the default is port 2000.
5. Run Carla
6. From the root of the repository run the training script:
   ```bash
   python train.py
   ```

Task Tracking:
[You can pick up a task from here](https://app.notion.com/p/aiea/2e0c03a14c2c80e7a552d722e9fc8df3?v=2e0c03a14c2c8020b4ec000c7b04f4ef&source=copy_link)
