"""
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

@author: Edward Denton
"""
import onnx
import onnxruntime
import onnxscript
import torch
from pathlib import Path
from train import load_params
from evaluate import load_model


def execute_onnx_runtime(model, example_input):
    onnx_inputs = onnx_inputs = [
        tensor.detach().cpu().numpy() for tensor in example_input
    ]

    print(f"Input length: {len(onnx_inputs)}")
    print(f"Sample input: {onnx_inputs}")

    ort_session = onnxruntime.InferenceSession(
        "../extension/model/AI_image_classifier_model.onnx", providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in
                         zip(ort_session.get_inputs(), onnx_inputs)}

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    with torch.no_grad():
        torch_outputs = model(*example_input)

    torch_outputs = torch_outputs.detach().cpu()
    onnxruntime_outputs = torch.from_numpy(onnxruntime_outputs)

    torch.testing.assert_close(
        torch_outputs,
        onnxruntime_outputs,
        rtol=1e-2,
        atol=1e-3,
    )

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")


def export_onnx(model):
    export_device = torch.device("cpu")

    model = model.to(export_device)
    model.eval()

    example_input = (torch.randn(1, 3, 224, 224, device=export_device),)
    onnx_program = torch.onnx.export(model, example_input, dynamo=True)

    onnx_program.save("../extension/model/AI_image_classifier_model.onnx")

    onnx_model = onnx.load("../extension/model/AI_image_classifier_model.onnx")
    onnx.checker.check_model(onnx_model)

    # https://netron.app/: Can be used to visualize the model

    execute_onnx_runtime(model, example_input)


def main():
    config = load_params()
    train_config = config["train"]
    data_config = config["data"]
    base_dir = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        architecture=train_config["architecture"],
        checkpoint_path=Path(base_dir / data_config["checkpoint_path"] / "best_model.pth"),
        device=device
    )

    export_onnx(model)


if __name__ == "__main__":
    main()
