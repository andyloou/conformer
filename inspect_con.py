import os
import sys
import argparse
import torch
from pytorch_nndct.apis import Inspector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    default="model_weights.pth",
    help='Converted Conformer .pth model file path')
parser.add_argument(
    '--target',
    default='DPUCZDX8G_ISA1_B4096',
    help='Specify target DPU device (e.g., DPUCZDX8G_ISA1_B4096)')
args = parser.parse_args()


def load_conformer_model(model_path):
    """Load the converted Conformer model"""
    from model import ConformerCTC
    
    print("Initializing Conformer model...")
    model = ConformerCTC()  # Use default configuration from model.py
    
    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    # Handle both direct state_dict and {'state_dict': ...} formats
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully!")
    return model


if __name__ == '__main__':
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    # Load model
    model = load_conformer_model(args.model_path)
    model = model.to(device)
    model.eval()

    # Dummy input: [batch, samples] for audio, [batch] for lengths
    batch_size = 1
    seq_length = 16000  # 1 second at 16kHz
    dummy_input_signal = torch.randn(batch_size, seq_length).to(device)  # [1, 16000]
    dummy_input_length = torch.tensor([seq_length], dtype=torch.int64).to(device)  # [1]

    # Inspect float model
    print(f"Inspecting model for target {args.target} ...")
    inspector = Inspector(args.target)
    inspector.inspect(model, (dummy_input_signal, dummy_input_length), device=device)
    print("Inspection completed.")