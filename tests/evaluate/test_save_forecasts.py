"""Unit tests for save_forecasts.py."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import xarray as xr


class MockDistribution:
    """Mock distribution that mimics the model output."""

    def __init__(self, mean: torch.Tensor) -> None:
        self._mean = mean

    @property
    def mean(self) -> torch.Tensor:
        return self._mean


class MockBatch:
    """Mock batch object that mimics ForecastingSample."""

    def __init__(self, zero_time: np.ndarray) -> None:
        self.zero_time = zero_time

    def to(self, device: str) -> "MockBatch":
        return self


class MockModel(torch.nn.Module):
    """Mock model that returns fake predictions."""

    def __init__(self, mean: torch.Tensor | None = None) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.mean = mean

    def forward(
        self, batch: MockBatch, grad_ckpt_start_timestep: int = 9999
    ) -> tuple[MockDistribution, list[Any]]:
        if self.mean is not None:
            return MockDistribution(self.mean), []
        mean = torch.randn(1, 5, 10, 2, 3)  # (batch, lat, lon, time, var)
        return MockDistribution(mean), []


class MockGriddedWeatherTask:
    """Mock GriddedWeatherTask."""

    def __init__(self) -> None:
        self.dataset = MagicMock()
        self.dataset.get_coord_by_name.side_effect = lambda name: MagicMock(
            values=np.linspace(-90, 90, 5)
            if name == "latitude"
            else np.linspace(0, 360, 10)
        )
        self.trg_variables_and_levels = ["2m_temperature"]
        self.temporal_resolution = np.timedelta64(6, "h")
        self.target_timedeltas = self.temporal_resolution * np.arange(1, 3)


def _create_mock_xr_dataset(time: np.datetime64) -> xr.Dataset:
    return xr.Dataset(
        {"temperature": (["time", "lat", "lon"], np.random.randn(1, 5, 10))},
        coords={
            "time": [time],
            "lat": np.arange(5),
            "lon": np.arange(10),
        },
    )


@patch("otter.evaluate.save_forecasts.forecast_to_xarray")
def test_generate_forecasts_creates_zarr(
    mock_forecast_to_xarray: MagicMock,
) -> None:
    """Test that generate_forecasts creates a zarr file."""
    from otter.evaluate.save_forecasts import generate_forecasts

    model = MockModel()
    model.eval()
    ds = MockGriddedWeatherTask()
    mock_forecast_to_xarray.return_value = _create_mock_xr_dataset(
        np.datetime64("2020-01-01")
    )

    dataloader = [MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"

        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
        )

        assert zarr_path.exists(), "Zarr file should be created"


@patch("otter.evaluate.save_forecasts.forecast_to_xarray")
def test_generate_forecasts_appends_batches(
    mock_forecast_to_xarray: MagicMock,
) -> None:
    """Test that multiple batches are appended to the zarr store."""
    from otter.evaluate.save_forecasts import generate_forecasts

    model = MockModel()
    model.eval()
    ds = MockGriddedWeatherTask()

    times = [
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-02"),
        np.datetime64("2020-01-03"),
    ]
    mock_forecast_to_xarray.side_effect = [
        _create_mock_xr_dataset(t) for t in times
    ]

    dataloader = [MockBatch(zero_time=np.array([t])) for t in times]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"

        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
        )

        result = xr.open_zarr(zarr_path)
        assert len(result.time) == 3, "Should have 3 time steps from 3 batches"


def test_generate_forecasts_saves_headline_fields_at_lead_frequency() -> None:
    from otter.evaluate.save_forecasts import generate_forecasts

    ds = MockGriddedWeatherTask()
    ds.trg_variables_and_levels = [
        "10m_u_component_of_wind",
        "geopotential/level_250",
        "geopotential/level_500",
        "temperature/level_500",
        "temperature/level_850",
        "total_column_water_vapour",
    ]
    ds.target_timedeltas = ds.temporal_resolution * np.arange(1, 9)
    # Each value encodes its lead index and field index.
    lead = torch.arange(8.0)[:, None]
    field = torch.arange(6.0)[None, :]
    mean = (100 * field + lead).expand(1, 5, 10, 8, 6)
    model = MockModel(mean)
    model.eval()
    dataloader = [MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"
        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
            headline_only=True,
            lead_frequency_hours=24,
        )
        result = xr.open_zarr(zarr_path).load()
        chunks = xr.open_zarr(zarr_path)["temperature"].encoding["chunks"]

    assert sorted(result.data_vars) == [
        "10m_u_component_of_wind",
        "geopotential",
        "temperature",
    ]
    assert list(result.level.values) == [500, 850]
    assert list(result.prediction_timedelta.values) == [
        np.timedelta64(24, "h"),
        np.timedelta64(48, "h"),
    ]
    at = dict(time=0, latitude=0, longitude=0)
    # Lead indices 3 and 7 are 24 and 48 hours.
    np.testing.assert_allclose(
        result["10m_u_component_of_wind"].isel(**at), [3, 7]
    )
    np.testing.assert_allclose(
        result["temperature"].isel(**at).sel(level=850), [403, 407]
    )
    # Geopotential is saved in its original units.
    np.testing.assert_allclose(
        result["geopotential"].isel(**at).sel(level=500), [203e3, 207e3]
    )
    assert chunks == (1, 5, 10, 2, 1)


def test_generate_forecasts_saves_ensemble_members() -> None:
    from otter.evaluate.save_forecasts import generate_forecasts

    class CountingModel(MockModel):
        """Each forward pass predicts the number of earlier passes."""

        def forward(
            self, batch: MockBatch, grad_ckpt_start_timestep: int = 9999
        ) -> tuple[MockDistribution, list[Any]]:
            count = float(self.dummy.data.item())
            self.dummy.data += 1
            return MockDistribution(torch.full((1, 5, 10, 2, 1), count)), []

    model = CountingModel()
    ds = MockGriddedWeatherTask()
    times = [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    dataloader = [MockBatch(zero_time=np.array([t])) for t in times]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"
        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
            num_ensemble_members=3,
        )
        result = xr.open_zarr(zarr_path).load()

    assert list(result.realization.values) == [0, 1, 2]
    # Three passes per init, the second init's after the first's.
    np.testing.assert_allclose(
        result["2m_temperature"]
        .isel(latitude=0, longitude=0, prediction_timedelta=0)
        .transpose("time", "realization"),
        [[0, 1, 2], [3, 4, 5]],
    )


def test_enable_mc_dropout_turns_on_only_dropout_and_drop_path() -> None:
    from functools import partial

    from otter.evaluate.save_forecasts import enable_mc_dropout
    from otter.models.architectures.mlp import SwiGLUMLP
    from otter.models.architectures.transformer_blocks import TransformerBlock

    block = TransformerBlock(
        token_dim=8,
        feedforward_network=partial(
            SwiGLUMLP, hidden_ratio=2.0, dropout_rate=0.5
        ),
        num_heads=2,
        dropout_rate=0.5,
        drop_path_rate=0.1,
        use_efficient_attention=False,
    )
    model = torch.nn.ModuleDict({"block": block}).eval()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        assert torch.equal(model["block"](x)[0], model["block"](x)[0])

        enable_mc_dropout(model)
        assert block.training and block.ffn.dropout.training
        assert not model.training and not block.mha_layer.training
        assert not torch.equal(model["block"](x)[0], model["block"](x)[0])


def test_mc_dropout_ensemble_size_reads_validation_config() -> None:
    from omegaconf import OmegaConf

    from otter.evaluate.save_forecasts import mc_dropout_ensemble_size

    def config(validation: dict[str, Any]) -> Any:
        return OmegaConf.create({"constants": {"validation": validation}})

    ensemble = config({"mc_dropout": True, "num_ensemble_members": 10})
    assert mc_dropout_ensemble_size(ensemble) == 10
    deterministic = config({"num_ensemble_members": 2})
    assert mc_dropout_ensemble_size(deterministic) is None
    assert (
        mc_dropout_ensemble_size(OmegaConf.create({"constants": {}})) is None
    )


def test_streamed_inference_preserves_autoregressive_means() -> None:
    """ compare both inference paths using the same stochastic samples """
    from types import MethodType, SimpleNamespace

    from otter.models.distributions import Delta, Gaussian
    from otter.models.forecasting_model import ForecastingModel

    for stochastic in (False, True):
        model = ForecastingModel.__new__(ForecastingModel)
        torch.nn.Module.__init__(model)
        model.data_grid_num_lat = 2
        model.data_grid_num_lon = 3
        model.temporal_resolution_hours = 6
        model.temporal_resolution_timedelta = np.timedelta64(6, "h")
        model.ctx_mean = torch.zeros(1, 1, 1, 1, 1)
        model.ctx_std = torch.ones(1, 1, 1, 1, 1)
        model.trg_mean = torch.full((1, 1, 1, 1, 1), 3.0)
        model.trg_std = torch.full((1, 1, 1, 1, 1), 2.0)
        model.idx_ctx_for_update = [0]

        def single_step(self, ctx, zero_time):
            mean = ctx[..., 0, :] + 0.5
            if stochastic:
                return Gaussian(mean, torch.zeros_like(mean), 0.1), None
            return Delta(mean), None

        model.single_time_forward = MethodType(single_step, model)
        batch = SimpleNamespace(
            ctx=torch.zeros(1, 2, 3, 2, 1),
            zero_time=np.array([np.datetime64("2020-07-01T00")]),
            trg_timedelta_hours=np.array([6, 18, 24], dtype=np.int32),
        )
        with torch.no_grad():
            torch.manual_seed(34)
            regular, _ = model(batch, grad_ckpt_start_timestep=9999)
            torch.manual_seed(34)
            streamed = list(model.iter_forecast_means(batch))

        assert [lead for lead, _ in streamed] == [6, 18, 24]
        stacked = torch.stack([frame for _, frame in streamed], dim=3)
        torch.testing.assert_close(stacked, regular.mean)


def test_streamed_saver_keeps_leads_and_realizations(tmp_path: Path) -> None:
    """ write each selected lead/member with the original output axes """
    from otter.evaluate.save_forecasts import generate_forecasts

    class StreamOnlyModel(MockModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *args, **kwargs):
            raise AssertionError("Streaming should not call full forward")

        def iter_forecast_means(self, batch):
            self.calls += 1
            for lead in (6, 12, 18, 24):
                mean = torch.full((1, 5, 10, 1), 10 * self.calls + lead)
                yield lead, mean

    ds = MockGriddedWeatherTask()
    ds.target_timedeltas = ds.temporal_resolution * np.arange(1, 5)
    model = StreamOnlyModel()
    batch = MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))
    output = tmp_path / "streamed.zarr"
    generate_forecasts(
        "test", model, [batch], ds, output,
        lead_frequency_hours=12,
        num_ensemble_members=3,
        stream_to_zarr=True,
    )

    with xr.open_zarr(output) as saved:
        saved = saved.load()
    assert model.calls == 3
    assert list(saved.realization.values) == [0, 1, 2]
    assert list(saved.prediction_timedelta.values) == [
        np.timedelta64(12, "h"), np.timedelta64(24, "h")
    ]
    actual = saved["2m_temperature"].isel(
        time=0, latitude=0, longitude=0,
    ).transpose("realization", "prediction_timedelta")
    np.testing.assert_array_equal(actual, [[22, 34], [32, 44], [42, 54]])


def test_direct_zarr_streaming_appends_dates_and_bounds_step_size(
    tmp_path: Path, monkeypatch,
) -> None:
    """ keep a one-lead write size with multiple dates and ensemble members """
    from otter.evaluate import save_forecasts

    class StreamOnlyModel(MockModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *args, **kwargs):
            raise AssertionError("Full rollout was unexpectedly requested")

        def iter_forecast_means(self, batch):
            self.calls += 1
            for lead in (6, 12, 18, 24):
                yield lead, torch.full(
                    (1, 5, 10, 1), self.calls * 100 + lead,
                )

    ds = MockGriddedWeatherTask()
    ds.target_timedeltas = ds.temporal_resolution * np.arange(1, 5)
    batches = [
        MockBatch(zero_time=np.array([np.datetime64("2020-01-01")])),
        MockBatch(zero_time=np.array([np.datetime64("2020-01-02")])),
    ]
    sizes = []
    convert = save_forecasts.forecast_to_xarray

    def limited_convert(*args, **kwargs):
        sizes.append(kwargs["forecast"].shape)
        return convert(*args, **kwargs)

    monkeypatch.setattr(save_forecasts, "forecast_to_xarray", limited_convert)
    model = StreamOnlyModel()
    output = tmp_path / "streamed_dates.zarr"
    save_forecasts.generate_forecasts(
        "test", model, batches, ds, output,
        num_ensemble_members=2, stream_to_zarr=True,
    )
    with xr.open_zarr(output) as stored:
        stored = stored.load()
    assert model.calls == 4
    assert len(sizes) == 16
    assert all(shape[3] == 1 for shape in sizes)
    assert stored.sizes["time"] == 2
    assert stored.sizes["realization"] == 2
    assert stored.sizes["prediction_timedelta"] == 4
    assert list(stored.time.values.astype("datetime64[D]")) == [
        np.datetime64("2020-01-01"), np.datetime64("2020-01-02"),
    ]
    actual = stored["2m_temperature"].isel(
        latitude=0, longitude=0,
    ).transpose("time", "realization", "prediction_timedelta")
    np.testing.assert_array_equal(actual[0], [
        [106, 112, 118, 124], [206, 212, 218, 224],
    ])
    np.testing.assert_array_equal(actual[1], [
        [306, 312, 318, 324], [406, 412, 418, 424],
    ])
    assert xr.open_zarr(output)["2m_temperature"].encoding["chunks"][-1] == 1


def test_direct_zarr_streaming_matches_original_output(tmp_path: Path) -> None:
    """ compare saved deterministic forecast fields and metadata """
    from otter.evaluate.save_forecasts import generate_forecasts

    class EquivalentModel(MockModel):
        def __init__(self):
            super().__init__()
            self.values = torch.arange(4.0).reshape(1, 1, 1, 4, 1)
            self.values = self.values.expand(1, 5, 10, 4, 1)

        def forward(self, batch, grad_ckpt_start_timestep=9999):
            return MockDistribution(self.values), []

        def iter_forecast_means(self, batch):
            for index, lead in enumerate((6, 12, 18, 24)):
                yield lead, self.values[..., index, :]

    ds = MockGriddedWeatherTask()
    ds.target_timedeltas = ds.temporal_resolution * np.arange(1, 5)
    batch = MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))
    original = tmp_path / "original.zarr"
    streamed = tmp_path / "direct.zarr"
    model = EquivalentModel()
    generate_forecasts("test", model, [batch], ds, original)
    generate_forecasts(
        "test", model, [batch], ds, streamed, stream_to_zarr=True,
    )
    with xr.open_zarr(original) as regular:
        regular = regular.load()
    with xr.open_zarr(streamed) as direct:
        direct = direct.load()
    xr.testing.assert_equal(direct, regular)
