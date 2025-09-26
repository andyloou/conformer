import torch
import torch.nn as nn
from collections import defaultdict

def inspect_checkpoint(checkpoint_path):
    """
    Inspect checkpoint to understand its structure including layer input/output shapes
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=" * 60)
    print("CHECKPOINT STRUCTURE INSPECTION")
    print("=" * 60)
    
    print("Checkpoint keys:", list(checkpoint.keys()))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\nNumber of parameters in state_dict: {len(state_dict)}")
        
        # Group parameters by layer
        layer_info = defaultdict(list)
        
        print("\n" + "=" * 50)
        print("DETAILED LAYER ANALYSIS")
        print("=" * 50)
        
        for key, tensor in state_dict.items():
            # Extract layer name (remove .weight, .bias, etc.)
            layer_name = key.rsplit('.', 1)[0] if '.' in key else key
            param_type = key.rsplit('.', 1)[1] if '.' in key else 'parameter'
            
            layer_info[layer_name].append({
                'param_name': key,
                'param_type': param_type,
                'shape': tensor.shape,
                'numel': tensor.numel(),
                'dtype': tensor.dtype
            })
        
        # Print layer information
        total_params = 0
        for layer_name, params in layer_info.items():
            print(f"\n📍 Layer: {layer_name}")
            print("-" * 40)
            
            layer_params = 0
            for param_info in params:
                layer_params += param_info['numel']
                print(f"  • {param_info['param_name']}")
                print(f"    - Shape: {param_info['shape']}")
                print(f"    - Type: {param_info['param_type']}")
                print(f"    - Elements: {param_info['numel']:,}")
                print(f"    - Data type: {param_info['dtype']}")
                
                # Infer layer type and input/output dimensions
                if param_info['param_type'] == 'weight':
                    shape = param_info['shape']
                    if len(shape) == 2:  # Linear layer
                        print(f"    - 🔍 LINEAR LAYER:")
                        print(f"      Input size: {shape[1]}")
                        print(f"      Output size: {shape[0]}")
                    elif len(shape) == 4:  # Conv2D layer
                        print(f"    - 🔍 CONV2D LAYER:")
                        print(f"      Output channels: {shape[0]}")
                        print(f"      Input channels: {shape[1]}")
                        print(f"      Kernel size: {shape[2]}x{shape[3]}")
                    elif len(shape) == 3:  # Conv1D or embedding
                        print(f"    - 🔍 CONV1D/EMBEDDING LAYER:")
                        print(f"      Dimension 0: {shape[0]}")
                        print(f"      Dimension 1: {shape[1]}")
                        print(f"      Dimension 2: {shape[2]}")
                    elif len(shape) == 5:  # Conv3D layer
                        print(f"    - 🔍 CONV3D LAYER:")
                        print(f"      Output channels: {shape[0]}")
                        print(f"      Input channels: {shape[1]}")
                        print(f"      Kernel size: {shape[2]}x{shape[3]}x{shape[4]}")
                
                print()
            
            total_params += layer_params
            print(f"  💾 Layer total parameters: {layer_params:,}")
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total parameters across all layers: {total_params:,}")
        print(f"Total number of layers: {len(layer_info)}")
        
        # Analyze model architecture
        print("\n🏗️  MODEL ARCHITECTURE ANALYSIS:")
        print("-" * 30)
        
        conv_layers = sum(1 for params in layer_info.values() 
                         for p in params if p['param_type'] == 'weight' and len(p['shape']) >= 3)
        linear_layers = sum(1 for params in layer_info.values() 
                           for p in params if p['param_type'] == 'weight' and len(p['shape']) == 2)
        
        print(f"• Convolutional layers: {conv_layers}")
        print(f"• Linear/Dense layers: {linear_layers}")
        
        # Find input and output dimensions
        first_layer_found = False
        last_layer_found = False
        
        sorted_layers = sorted(layer_info.items())
        
        if sorted_layers:
            # Try to find first layer (input)
            for layer_name, params in sorted_layers:
                weight_param = next((p for p in params if p['param_type'] == 'weight'), None)
                if weight_param and not first_layer_found:
                    shape = weight_param['shape']
                    if len(shape) >= 2:
                        if len(shape) == 2:  # Linear
                            print(f"• Estimated input size: {shape[1]}")
                        elif len(shape) == 4:  # Conv2D
                            print(f"• Estimated input channels: {shape[1]}")
                        first_layer_found = True
                        break
            
            # Try to find last layer (output)
            for layer_name, params in reversed(sorted_layers):
                weight_param = next((p for p in params if p['param_type'] == 'weight'), None)
                if weight_param and not last_layer_found:
                    shape = weight_param['shape']
                    if len(shape) >= 2:
                        if len(shape) == 2:  # Linear
                            print(f"• Estimated output size: {shape[0]}")
                        elif len(shape) == 4:  # Conv2D
                            print(f"• Estimated output channels: {shape[0]}")
                        last_layer_found = True
                        break
    
    if 'hyper_parameters' in checkpoint:
        print("\n" + "=" * 50)
        print("HYPERPARAMETERS")
        print("=" * 50)
        hyper_params = checkpoint['hyper_parameters']
        for key, value in hyper_params.items():
            print(f"• {key}: {value}")
    
    # Additional checkpoint information
    print("\n" + "=" * 50)
    print("ADDITIONAL CHECKPOINT INFO")
    print("=" * 50)
    
    for key in checkpoint.keys():
        if key not in ['state_dict', 'hyper_parameters']:
            value = checkpoint[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"• {key}: {value}")
            elif isinstance(value, torch.Tensor):
                print(f"• {key}: Tensor {list(value.shape)} - {value.numel():,} elements - {value.dtype}")
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                print(f"• {key}: {value}")
            else:
                try:
                    size = len(value) if hasattr(value, '__len__') else 'N/A'
                    print(f"• {key}: {type(value)} (size: {size})")
                except TypeError:
                    print(f"• {key}: {type(value)} (scalar or special type)")

def analyze_model_from_checkpoint(checkpoint_path, sample_input_shape=None):
    """
    Additional function to create a model from checkpoint and analyze it
    """
    print("\n" + "=" * 60)
    print("ATTEMPTING MODEL RECONSTRUCTION")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            # Try to create a simple sequential model to test
            if sample_input_shape:
                print(f"Testing with sample input shape: {sample_input_shape}")
                sample_input = torch.randn(1, *sample_input_shape)
                print(f"Sample input created: {sample_input.shape}")
        
    except Exception as e:
        print(f"Could not reconstruct model: {e}")

# Main execution
if __name__ == "__main__":
    checkpoint_path = "model_weights.ckpt"
    
    print("🔍 Inspecting checkpoint structure...")
    inspect_checkpoint(checkpoint_path)
    
    # Optional: Try to analyze with sample input
    # Uncomment and modify the shape based on your model
    # analyze_model_from_checkpoint(checkpoint_path, sample_input_shape=(3, 224, 224))  # For image models
    # analyze_model_from_checkpoint(checkpoint_path, sample_input_shape=(768,))  # For text models