import argparse
import importlib
from pathlib import Path

import addict
import torch

from birdclef.models import builder as models_builder
from birdclef import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weights')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 640], help='height width')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_names', nargs='+', default=['input_0'])
    parser.add_argument('--output_names', nargs='+', default=['output_0'])
    parser.add_argument('--checkpoint_key', default=None)
    parser.add_argument('--opset_version', type=int, default=11)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_path = Path(args.config).absolute()
    weights_path = Path(args.weights).absolute()
    image_size = args.image_size
    batch_size = args.batch_size
    input_names = args.input_names
    output_names = args.output_names
    checkpoint_key = args.checkpoint_key
    opset_version = args.opset_version
    verbose = args.verbose

    config_module_path = '.'.join([config_path.parent.name, config_path.stem])
    config = importlib.import_module(str(config_module_path))

    model_cfg = addict.Dict(config.model)

    model = models_builder.build_model(model_cfg)
    # utils.load_checkpoint(model, str(weights_path), checkpoint_key)

    model = model.eval()

    inputs = torch.randn(batch_size, 3, *image_size, requires_grad=False)
    torch_out = model(inputs)

    onnx_filename = str(weights_path.name + '.onnx')

    torch.onnx.export(
        model,
        inputs,
        onnx_filename,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: 'batch_size'}},
        verbose=verbose
    )


if __name__ == '__main__':
    main()
