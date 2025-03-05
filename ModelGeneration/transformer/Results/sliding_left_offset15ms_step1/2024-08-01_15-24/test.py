import torch

# Load the .pth file
file_path = r'C:\Users\marco\master_project\humanObjectDetection\ModelGeneration\transformer\Results\sliding_left_offset15ms_step1\2024-08-01_15-24\checkpoints\model_best.pth'
model_data = torch.load(file_path, map_location=torch.device('cpu'))

# Check if the file contains a model state_dict or an entire model object
if isinstance(model_data, dict):
    if 'state_dict' in model_data:
        # Print information from the state_dict
        print("Model's state_dict:")
        for key, value in model_data['state_dict'].items():
            print(f"{key}: {value.shape}")
    else:
        # Assume it's a state_dict directly
        print("Model's state_dict:")
        for key, value in model_data.items():
            print(f"{key}: {value.shape}")
else:
    # Assume the entire model was saved
    print("Entire model object loaded.")
    print(model_data)

# Print additional metadata if available
if isinstance(model_data, dict):
    
    if 'epoch' in model_data:
        print(f"\nEpoch: {model_data['epoch']}")

 
  
