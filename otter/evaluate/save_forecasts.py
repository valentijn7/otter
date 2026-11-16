# %%
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from otter.data.datasets import GriddedWeatherTask
from otter.data.labelled_array import forecast_to_xarray
from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS
from otter.evaluate.constants import HEADLINE_LEVELS, HEADLINE_VARIABLES
from otter.experiment.run_state import RunState
from otter.experiment.utils import load_experiment_config
from otter.models.architectures.transformer_blocks import TransformerBlock
from otter.models.forecasting_model import ForecastingModel

# %%
logging.basicConfig(level=logging.INFO)


def load_dependencies(
    experiment_name: str,
    overrides: list[str] | None = None,
    root_results: str = "_results",
    data_path: str | None = None,
):
    if overrides is None:
        overrides = []

    config = load_experiment_config(
        experiment_name,
        override_args=overrides,
        results_dir=root_results,
    )
    config.model.device = "cuda"
    # The saved paths are those of the machine that trained the run.
    config.info.experiment_path = f"{root_results}/{experiment_name}"
    if data_path is not None:
        config.data_source.path = data_path
    config.test_dataloader.dataset.load_targets = False

    experiment = instantiate(config)
    experiment_path = Path(experiment.info.experiment_path)

    dataloader = experiment.test_dataloader
    ds: GriddedWeatherTask = dataloader.dataset

    model: ForecastingModel = experiment.model.to("cuda")

    checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
    run_state = RunState.load(checkpoint_path)
    model.load_state_dict(run_state.model_state)
    model.eval()
    if mc_dropout_ensemble_size(config) is not None:
        enable_mc_dropout(model)

    zarr_path = experiment_path / "forecasts.zarr"

    return (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    )


def mc_dropout_ensemble_size(config: DictConfig) -> int | None:
    """Size of the MC dropout ensemble a run was validated with, or None
    for a deterministic run.

    Runs fine-tuned with a CRPS loss on MC dropout ensembles only forecast
    well with dropout on: with it off, they drift.
    """
    validation = config.constants.get("validation", {})
    if not validation.get("mc_dropout", False):
        return None
    return int(validation.num_ensemble_members)


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Turn on dropout and drop-path, so that each forward pass samples an
    MC dropout ensemble member, as in the runs' validation."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, TransformerBlock)):
            # Not .train(), which would also put submodules, such as
            # mixture-of-experts layers, in train mode.
            module.training = True


def headline_fields(variables_and_levels: list[str]) -> list[int]:
    """Indices of the headline variables, at every headline level."""
    fields = []
    for index, name in enumerate(variables_and_levels):
        variable, _, level = name.partition("/level_")
        if variable in HEADLINE_VARIABLES and (
            not level or int(level) in HEADLINE_LEVELS
        ):
            fields.append(index)
    return fields


def _resize_dimension(group: zarr.Group, dimension: str, size: int) -> None:
    """ resize only arrays whose xarray dimensions include the named axis """
    for name in group.array_keys():
        array = group[name]
        dimensions = list(array.attrs["_ARRAY_DIMENSIONS"])
        if dimension in dimensions:
            shape = list(array.shape)
            shape[dimensions.index(dimension)] = size
            array.resize(tuple(shape))


def _write_forecast_region(
    zarr_path: Path,
    group: Any,
    step: xr.Dataset,
    lead_index: int,
    selected_leads: list[int],
    member_index: int | None,
    num_members: int | None,
    time_offset: int,
) -> Any:
    """ write one forecast lead/member directly to the final Zarr layout """
    if group is None:
        encoding: dict[str, dict[str, Any]] = {
            "time": {
                "units": "hours since 2020-01-01T00:00:00",
                "dtype": "int64",
            },
            "prediction_timedelta": {"units": "hours", "dtype": "int64"},
        }
        for name, field in step.data_vars.items():
            encoding[name] = {
                "chunks": [
                    1 if dim in (
                        "time", "realization", "prediction_timedelta", "level"
                    ) else size
                    for dim, size in field.sizes.items()
                ]
            }
        # Defer consolidation: stale metadata otherwise hides later resizes.
        step.to_zarr(
            zarr_path, mode="w", encoding=encoding,
            zarr_format=2, consolidated=False,
        )
        group = zarr.open_group(
            str(zarr_path), mode="a", use_consolidated=False,
        )

        # Resize each array once: separate resizes of lead and realization
        # can restore the other's old size through cached array metadata.
        for name in group.array_keys():
            array = group[name]
            dimensions = list(array.attrs["_ARRAY_DIMENSIONS"])
            shape = list(array.shape)
            if "prediction_timedelta" in dimensions:
                shape[dimensions.index("prediction_timedelta")] = (
                    len(selected_leads)
                )
            if num_members is not None and "realization" in dimensions:
                shape[dimensions.index("realization")] = num_members
            if tuple(shape) != array.shape:
                array.resize(tuple(shape))

        # The CF encodings above store timedelta coordinates in hours.
        group["prediction_timedelta"][:] = selected_leads
        if num_members is not None:
            group["realization"][:] = np.arange(num_members)
        return group

    for name, field in step.data_vars.items():
        region = []
        for dimension in field.dims:
            if dimension == "time":
                region.append(
                    slice(time_offset, time_offset + field.sizes["time"])
                )
            elif dimension == "prediction_timedelta":
                region.append(slice(lead_index, lead_index + 1))
            elif dimension == "realization":
                if member_index is None:
                    raise ValueError("Missing ensemble member index")
                region.append(slice(member_index, member_index + 1))
            else:
                region.append(slice(None))
        group[name][tuple(region)] = field.values
    return group


def _stream_batch_to_zarr(
    model: ForecastingModel,
    batch: Any,
    zero_times: npt.NDArray[Any],
    zarr_path: Path,
    selected_leads: list[int],
    fields: list[int],
    variables_and_levels: list[str],
    temporal_resolution_hours: int,
    lats: npt.NDArray[Any],
    lons: npt.NDArray[Any],
    num_members: int | None,
    time_offset: int,
    group: Any,
) -> Any:
    """ stream each selected lead of each member without accumulating them """
    positions = {lead: index for index, lead in enumerate(selected_leads)}
    for member in range(num_members or 1):
        seen = set()
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=torch.float16),
        ):
            for lead, mean in model.iter_forecast_means(batch):
                if lead not in positions:
                    continue
                frame = mean[..., fields].detach().cpu().numpy()
                step = forecast_to_xarray(
                    forecast=frame[..., None, :],
                    zero_times=zero_times,
                    temporal_resolution_hours=temporal_resolution_hours,
                    variables_and_levels=variables_and_levels,
                    lats=lats,
                    lons=lons,
                    lead_time_hours=[lead],
                )
                if num_members is not None:
                    step = step.expand_dims(realization=[member])
                for variable, factor in VARIABLES_WITH_SCALING_FACTORS.items():
                    if variable in step:
                        step[variable] = step[variable] / factor
                group = _write_forecast_region(
                    zarr_path, group, step, positions[lead], selected_leads,
                    member if num_members is not None else None,
                    num_members, time_offset,
                )
                seen.add(lead)
                del frame, step
        if seen != set(selected_leads):
            raise ValueError("Missing selected leads in streamed forecast")
    return group


def _append_streamed_times(
    group: Any,
    zero_times: npt.NDArray[Any],
    time_offset: int,
) -> None:
    """ extend the existing Zarr store for another initialization batch """
    count = len(zero_times)
    _resize_dimension(group, "time", time_offset + count)
    offset_hours = (
        zero_times.astype("datetime64[ns]")
        - np.datetime64("2020-01-01T00:00:00", "ns")
    ) / np.timedelta64(1, "h")
    if not np.all(offset_hours == np.floor(offset_hours)):
        raise ValueError("Forecast init time is not aligned to the hour")
    group["time"][time_offset:time_offset + count] = offset_hours.astype(int)


def generate_forecasts(
    experiment_name: str,
    model: ForecastingModel,
    dataloader: torch.utils.data.DataLoader,
    ds: GriddedWeatherTask,
    zarr_path: Path,
    headline_only: bool = False,
    lead_frequency_hours: int | None = None,
    num_ensemble_members: int | None = None,
    stream_to_zarr: bool = False,
):
    """Forecast from each init in the dataloader, and save to zarr.

    With num_ensemble_members set, the model is run that many times per
    init, and the members are saved along a "realization" dimension. The
    model should then sample, e.g. with enable_mc_dropout.

    stream_to_zarr writes one selected lead/member at a time to Zarr.
    It bounds GPU and CPU output memory independent of horizon and members.
    Training and the default inference path remain unchanged.
    """
    lats = ds.dataset.get_coord_by_name("latitude").values
    lons = ds.dataset.get_coord_by_name("longitude").values

    temporal_resolution_hours = int(
        ds.temporal_resolution / np.timedelta64(1, "h")
    )
    fields = list(range(len(ds.trg_variables_and_levels)))
    if headline_only:
        fields = headline_fields(ds.trg_variables_and_levels)
    variables_and_levels = [ds.trg_variables_and_levels[i] for i in fields]

    all_lead_time_hours = [
        int(lead / np.timedelta64(1, "h")) for lead in ds.target_timedeltas
    ]
    leads = [
        index
        for index, hours in enumerate(all_lead_time_hours)
        if lead_frequency_hours is None or hours % lead_frequency_hours == 0
    ]
    lead_time_hours = [all_lead_time_hours[i] for i in leads]

    if not lead_time_hours:
        raise ValueError("No selected forecast lead times")
    if stream_to_zarr and zarr_path.exists():
        raise FileExistsError(zarr_path)
    group = None
    time_offset = 0

    # Loop through the dataloader and save to Zarr
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        zero_times = batch.zero_time
        print(f"Generating forecast with zero_time {zero_times}")
        batch = batch.to("cuda")
        if stream_to_zarr:
            if batch_idx:
                _append_streamed_times(group, zero_times, time_offset)
            group = _stream_batch_to_zarr(
                model, batch, zero_times, zarr_path, lead_time_hours,
                fields, variables_and_levels, temporal_resolution_hours,
                lats, lons, num_ensemble_members, time_offset, group,
            )
            time_offset += len(zero_times)
            continue
        members = []
        for _ in range(num_ensemble_members or 1):
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", dtype=torch.float16),
            ):
                prd_dist, _ = model(
                    batch=batch, grad_ckpt_start_timestep=9999
                )
                prediction = prd_dist.mean[:, :, :, leads][..., fields]
                forecast = prediction.cpu().numpy()
            members.append(
                forecast_to_xarray(
                    forecast=forecast,
                    zero_times=zero_times,
                    temporal_resolution_hours=temporal_resolution_hours,
                    variables_and_levels=variables_and_levels,
                    lats=lats,
                    lons=lons,
                    lead_time_hours=lead_time_hours,
                )
            )

        if num_ensemble_members is None:
            (ds_batch,) = members
        else:
            # The dimension name WeatherBench2 expects for ensembles.
            ds_batch = xr.concat(members, dim="realization")
            ds_batch = ds_batch.assign_coords(
                realization=np.arange(num_ensemble_members)
            )

        for variable, scaling_factor in VARIABLES_WITH_SCALING_FACTORS.items():
            if variable in ds_batch:
                # Undo the scaling for some variables to adjust units.
                # This scaling doesn't apply to the zarr dataset directly, so we
                # need to reverse it here.
                ds_batch[variable] = ds_batch[variable] / scaling_factor

        if batch_idx == 0:
            # Initialize the zarr store on disk
            encoding: dict[str, dict[str, Any]] = {
                "time": {
                    "units": "hours since 2020-01-01T00:00:00",
                    "dtype": "int64",
                }
            }
            # One chunk per field per forecast, rather than many small
            # chunks, which are slow to write to blob storage.
            for name, field in ds_batch.data_vars.items():
                chunks = [
                    1 if dim in ("time", "level", "realization") else size
                    for dim, size in field.sizes.items()
                ]
                encoding[str(name)] = {"chunks": chunks}
            ds_batch.to_zarr(
                zarr_path, mode="w", encoding=encoding, zarr_format=2
            )
        else:
            # Append to the existing zarr store along the 'time' dimension
            ds_batch.to_zarr(zarr_path, append_dim="time", zarr_format=2)

    if stream_to_zarr and group is not None:
        # Resizing metadata must be consolidated after the final write.
        zarr.consolidate_metadata(str(zarr_path))


# %%
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="Hydra config overrides",
        default=None,
    )
    parser.add_argument(
        "--root_results",
        default="_results",
        help="Directory holding the experiment. Forecasts are saved in it.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Overrides data_source.path from the experiment config.",
    )
    parser.add_argument(
        "--headline_only",
        action="store_true",
        help="Save only the variables and levels of the headline scores.",
    )
    parser.add_argument(
        "--lead_frequency_hours",
        type=int,
        default=None,
        help="Save only lead times that are multiples of this.",
    )
    parser.add_argument(
        "--stream_to_zarr", action="store_true",
        help="Write each forecast lead/member to Zarr immediately.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name

    (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    ) = load_dependencies(
        experiment_name,
        overrides=args.overrides,
        root_results=args.root_results,
        data_path=args.data_path,
    )

    if not zarr_path.exists():
        logging.info(f"Generating forecasts for {experiment_name}...")
        generate_forecasts(
            experiment_name=experiment_name,
            model=model,
            dataloader=dataloader,
            ds=ds,
            zarr_path=zarr_path,
            headline_only=args.headline_only,
            lead_frequency_hours=args.lead_frequency_hours,
            num_ensemble_members=mc_dropout_ensemble_size(config),
            stream_to_zarr=args.stream_to_zarr,
        )
    else:
        logging.info(
            f"Forecast already exists at {zarr_path}, skipping generation."
        )
