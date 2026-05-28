import json
import torch
from model import LSTMWordPredictor

def main():
    print("Loading configuration...")
    with open("processed_dataset.json", "r") as f:
        data = json.load(f)
        
    vocab_size = data["vocab_size"]
    context_length = data["context_length"]
    
    # Initialize model with exact same parameters as training
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    
    model = LSTMWordPredictor(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    print("Loading PyTorch model state dict...")
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()
    
    # Create dummy input of shape [batch_size, sequence_length]
    # batch_size=1, sequence_length=context_length
    dummy_input = torch.zeros((1, context_length), dtype=torch.long)
    
    onnx_path = "../app/lstm_model.onnx"
    print(f"Exporting model to ONNX at {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    
    # Keep a copy locally in training folder as well
    torch.onnx.export(
        model,
        dummy_input,
        "lstm_model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    
    print("ONNX Model exported successfully.")
    
    # Attempt Dynamic Quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print("onnxruntime.quantization found. Quantizing model to INT8...")
        
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path,  # Overwrite with quantized version
            weight_type=QuantType.QUInt8
        )
        print("Quantized ONNX model saved to the app folder successfully.")
    except Exception as e:
        print(f"Could not perform quantization (possibly due to missing dependencies): {e}")
        print("Continuing with standard FP32 ONNX model.")

if __name__ == "__main__":
    main()
