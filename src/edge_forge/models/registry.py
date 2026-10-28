"""Model registry, lightweight reference architectures and checkpoint loading.

The registry provides deterministic, dependency-free computer-vision backbones
that are small enough to export and quantize in seconds. They are used by the
test-suite, by the CLI demo commands and by AI agents that need a reproducible
subject under test. Real checkpoints can be loaded from TorchScript files, full
``nn.Module`` pickles or state dictionaries paired with a registered
architecture.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from edge_forge.errors import ConfigurationError, ModelLoadError, ModelNotFoundError
from edge_forge.models.schemas import ModelInfo, Precision, TensorSpec


class ConvBlock(nn.Module):
    """Convolution, batch normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(self.norm(self.conv(inputs))))


class EdgeCNN(nn.Module):
    """Compact convolutional backbone suitable for microcontrollers and SBCs."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        width: int = 16,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 1 or depth < 1:
            raise ConfigurationError("EdgeCNN width and depth must be positive")
        layers: list[nn.Module] = [
            ConvBlock(input_channels, width),
            ConvBlock(width, width, stride=2),
        ]
        channels = width
        for _ in range(depth - 1):
            next_channels = min(channels * 2, width * 8)
            layers.append(ConvBlock(channels, next_channels, stride=2))
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs)).flatten(1)
        return cast(torch.Tensor, self.classifier(self.dropout(features)))


class DepthwiseBlock(nn.Module):
    """Depthwise separable convolution block with a residual connection."""

    def __init__(self, channels: int, expansion: int = 2, stride: int = 1) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm_depthwise = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm_pointwise = nn.BatchNorm2d(hidden)
        self.projection = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm_projection = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU6(inplace=True)
        self.use_residual = stride == 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.norm_depthwise(self.depthwise(inputs)))
        hidden = self.activation(self.norm_pointwise(self.pointwise(hidden)))
        hidden = self.norm_projection(self.projection(hidden))
        if self.use_residual:
            return cast(torch.Tensor, inputs + hidden)
        return cast(torch.Tensor, hidden)


class MobileEdgeNet(nn.Module):
    """MobileNet-inspired depthwise separable network for edge inference."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        width: int = 16,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 1 or depth < 1:
            raise ConfigurationError("MobileEdgeNet width and depth must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU6(inplace=True),
        )
        blocks: list[nn.Module] = []
        channels = width
        for index in range(depth):
            stride = 2 if index in (1, depth - 1) and index != 0 else 1
            blocks.append(DepthwiseBlock(channels, expansion=2, stride=stride))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.blocks(self.stem(inputs))).flatten(1)
        return cast(torch.Tensor, self.classifier(self.dropout(features)))


class AttentionBlock(nn.Module):
    """Pre-norm transformer block with explicit attention math for ONNX export."""

    def __init__(self, width: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        if width % heads != 0:
            raise ConfigurationError("attention width must be divisible by the number of heads")
        self.heads = heads
        self.scale = (width // heads) ** -0.5
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = inputs.shape
        normalized = self.norm_attention(inputs)
        qkv = self.qkv(normalized)
        query, key, value = qkv.chunk(3, dim=-1)
        head_dim = width // self.heads
        query = query.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(batch, tokens, width)
        inputs = inputs + self.projection(context)
        return cast(torch.Tensor, inputs + self.mlp(self.norm_mlp(inputs)))


class TinyViT(nn.Module):
    """Small vision transformer with fixed spatial resolution and dynamic batch."""

    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        num_classes: int = 10,
        width: int = 32,
        depth: int = 2,
        heads: int = 4,
        patch_size: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ConfigurationError("input_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.input_size = input_size
        self.patch_embed = nn.Conv2d(
            input_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        token_count = (input_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.positional = nn.Parameter(torch.zeros(1, token_count + 1, width))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        batch = patches.shape[0]
        tokens = torch.cat(
            [self.class_token.expand(batch, -1, -1), patches],
            dim=1,
        )
        tokens = tokens + self.positional
        for block in self.blocks:
            tokens = block(tokens)
        return cast(torch.Tensor, self.classifier(self.dropout(self.norm(tokens)[:, 0])))


Builder = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelBlueprint:
    """Registry entry describing one reference architecture."""

    name: str
    family: str
    description: str
    builder: Builder
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    supports_dynamic_spatial: bool = True


class ModelRegistry:
    """Registry of named model builders with validated keyword arguments."""

    def __init__(self) -> None:
        self._blueprints: dict[str, ModelBlueprint] = {}

    def register(self, blueprint: ModelBlueprint) -> None:
        """Register a blueprint, rejecting duplicate names."""
        if blueprint.name in self._blueprints:
            raise ConfigurationError(f"model '{blueprint.name}' is already registered")
        self._blueprints[blueprint.name] = blueprint

    def get(self, name: str) -> ModelBlueprint:
        """Return a blueprint by name."""
        try:
            return self._blueprints[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._blueprints))
            raise ModelNotFoundError(
                f"unknown model '{name}'; available models: {available}"
            ) from exc

    def names(self) -> list[str]:
        """Return every registered model name sorted alphabetically."""
        return sorted(self._blueprints)

    def blueprints(self) -> list[ModelBlueprint]:
        """Return every blueprint sorted by name."""
        return [self._blueprints[name] for name in self.names()]

    def build(self, name: str, **overrides: Any) -> nn.Module:
        """Instantiate a model, merging defaults with caller overrides.

        Reference architectures are initialized deterministically by default so
        that an exported artifact can be audited later by rebuilding the same
        model. Pass ``seed=None`` to opt into non-deterministic initialization.
        """
        blueprint = self.get(name)
        kwargs = {**blueprint.default_kwargs, **overrides}
        seed = kwargs.pop("seed", 0)
        try:
            if seed is None:
                model = blueprint.builder(**kwargs)
            else:
                state = torch.random.get_rng_state()
                try:
                    torch.manual_seed(int(seed))
                    model = blueprint.builder(**kwargs)
                finally:
                    torch.random.set_rng_state(state)
        except TypeError as exc:
            raise ConfigurationError(f"invalid arguments for model '{name}': {exc}") from exc
        if not isinstance(model, nn.Module):
            raise ConfigurationError(f"builder for '{name}' did not return an nn.Module")
        return model.eval()


REGISTRY = ModelRegistry()
REGISTRY.register(
    ModelBlueprint(
        name="edge_cnn",
        family="cnn",
        description="Three-stage convolutional classifier with adaptive pooling.",
        builder=EdgeCNN,
        default_kwargs={"width": 16, "depth": 3, "num_classes": 10},
    )
)
REGISTRY.register(
    ModelBlueprint(
        name="mobile_edge_net",
        family="depthwise-cnn",
        description="MobileNet-style depthwise separable classifier for embedded CPUs.",
        builder=MobileEdgeNet,
        default_kwargs={"width": 16, "depth": 4, "num_classes": 10},
    )
)
REGISTRY.register(
    ModelBlueprint(
        name="tiny_vit",
        family="transformer",
        description="Two-block vision transformer with a fixed 32x32 spatial resolution.",
        builder=TinyViT,
        default_kwargs={"width": 32, "depth": 2, "heads": 4, "num_classes": 10},
        supports_dynamic_spatial=False,
    )
)


def build_model(name: str, **overrides: Any) -> nn.Module:
    """Build a registered model in evaluation mode."""
    return REGISTRY.build(name, **overrides)


def get_blueprint(name: str) -> ModelBlueprint:
    """Return the registry entry for a model name."""
    return REGISTRY.get(name)


def list_models() -> list[ModelBlueprint]:
    """Return every registered model blueprint."""
    return REGISTRY.blueprints()


def describe_models() -> list[dict[str, Any]]:
    """Return a JSON-serializable catalog of registered models."""
    catalog: list[dict[str, Any]] = []
    for blueprint in REGISTRY.blueprints():
        model = REGISTRY.build(blueprint.name)
        parameters = count_parameters(model)
        catalog.append(
            {
                "name": blueprint.name,
                "family": blueprint.family,
                "description": blueprint.description,
                "parameters": parameters,
                "size_mb": estimate_model_size_mb(model),
                "default_kwargs": dict(blueprint.default_kwargs),
                "supports_dynamic_spatial": blueprint.supports_dynamic_spatial,
            }
        )
    return catalog


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate the serialized FP32 size of a model in mebibytes."""
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    total_bytes += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return round(total_bytes / (1024 * 1024), 4)


def make_dummy_input(shape: tuple[int, ...], *, seed: int | None = None) -> torch.Tensor:
    """Create a deterministic random input tensor."""
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ConfigurationError("dummy input shape must contain positive dimensions")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def extract_output_tensors(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in output):
        return list(output)
    if isinstance(output, dict) and all(isinstance(item, torch.Tensor) for item in output.values()):
        return list(output.values())
    raise ModelLoadError("model forward pass returned an unsupported output structure")


def inspect_torch_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    name: str = "torch_model",
    architecture: str | None = None,
    seed: int = 0,
) -> ModelInfo:
    """Collect parameters, size and tensor specifications from a PyTorch model."""
    model.eval()
    dummy = make_dummy_input(input_shape, seed=seed)
    with torch.inference_mode():
        outputs = extract_output_tensors(model(dummy))
    output_specs = [
        TensorSpec(
            name=f"output_{index}",
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
        )
        for index, tensor in enumerate(outputs)
    ]
    return ModelInfo(
        name=name,
        architecture=architecture or type(model).__name__,
        parameters=count_parameters(model),
        size_mb=estimate_model_size_mb(model),
        input_specs=[
            TensorSpec(name="input", shape=input_shape, dtype="float32"),
        ],
        output_specs=output_specs,
        precision=Precision.FP32,
        metadata={"family": "torch"},
    )


@dataclass(frozen=True)
class LoadedModel:
    """A PyTorch model together with the metadata needed to export it."""

    model: nn.Module
    architecture: str
    input_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    architecture: str,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a state dictionary with the metadata required to rebuild it."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "model_kwargs": dict(model_kwargs or {}),
    }
    torch.save(payload, path)
    return path


def _load_torchscript(path: Path) -> nn.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = torch.jit.load(str(path), map_location="cpu")  # type: ignore[no-untyped-call]
            return cast(nn.Module | None, loaded)
    except Exception:
        return None


def load_model(
    path: str | Path,
    *,
    architecture: str | None = None,
    input_shape: tuple[int, ...] | None = None,
    seed: int = 0,
    allow_unsafe: bool = False,
) -> LoadedModel:
    """Load a registered state dict or, opt-in, a TorchScript module or pickle.

    State dictionaries are loaded with PyTorch's restricted unpickler. TorchScript
    archives and full ``nn.Module`` pickles can execute arbitrary code during
    deserialization, so both require ``allow_unsafe=True`` from a trusted caller.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ModelNotFoundError(f"model file does not exist: {resolved}")
    if resolved.suffix.lower() == ".onnx":
        raise ModelLoadError(
            "load_model handles PyTorch checkpoints only; use inspect_onnx for ONNX files"
        )

    scripted = _load_torchscript(resolved) if allow_unsafe else None
    if scripted is not None:
        scripted.eval()
        shape = input_shape or (1, 3, 32, 32)
        return LoadedModel(
            model=scripted,
            architecture="torchscript",
            input_shape=shape,
            metadata={"source": str(resolved), "format": "torchscript"},
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = torch.load(str(resolved), map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe:
            raise ModelLoadError(
                "checkpoint is not a state dictionary; loading a TorchScript archive or a "
                "full module pickle requires allow_unsafe=True and a trusted file"
            ) from safe_error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelLoadError(f"unable to load checkpoint {resolved}: {exc}") from exc

    if isinstance(payload, nn.Module):
        payload.eval()
        shape = input_shape or (1, 3, 32, 32)
        return LoadedModel(
            model=payload,
            architecture=type(payload).__name__,
            input_shape=shape,
            metadata={"source": str(resolved), "format": "module"},
        )

    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        resolved_architecture = str(payload.get("architecture") or architecture or "")
        model_kwargs = dict(payload.get("model_kwargs") or {})
        shape = input_shape or tuple(payload.get("input_shape") or (1, 3, 32, 32))
    else:
        state_dict = payload
        resolved_architecture = architecture or ""
        model_kwargs = {}
        shape = input_shape or (1, 3, 32, 32)

    if not resolved_architecture:
        raise ModelLoadError(
            "checkpoint does not embed an architecture; pass architecture= explicitly"
        )
    model = build_model(resolved_architecture, **model_kwargs)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(
            f"state dict is incompatible with architecture '{resolved_architecture}': {exc}"
        ) from exc
    model.eval()
    return LoadedModel(
        model=model,
        architecture=resolved_architecture,
        input_shape=shape,
        metadata={"source": str(resolved), "format": "state_dict", "seed": seed},
    )
