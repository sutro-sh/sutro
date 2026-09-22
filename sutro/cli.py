from datetime import timezone
import json
import os

import click
import requests
from colorama import Fore, Style
from sutro.sdk import Sutro
from sutro.validation import (
    API_KEY_ENV,
    API_URL_ENV,
    load_config,
    normalize_api_url,
    resolve_api_configuration,
    resolve_api_configuration_with_context,
    save_config,
)
import polars as pl
import warnings

warnings.filterwarnings("ignore", category=pl.PolarsInefficientMapWarning)
pl.Config.set_tbl_hide_dataframe_shape(True)


def check_auth():
    api_key, api_url = resolve_api_configuration()
    return api_key is not None and api_url is not None


def get_sdk():
    return Sutro()


def set_config_api_url(api_url: str):
    config = load_config()
    normalized_api_url = normalize_api_url(api_url)
    current_api_url = config.get("api_url") or config.get("base_url")
    try:
        normalized_current_api_url = (
            normalize_api_url(current_api_url) if current_api_url else None
        )
    except ValueError:
        normalized_current_api_url = None

    deployment_changed = normalized_current_api_url != normalized_api_url
    config["api_url"] = normalized_api_url
    config.pop("base_url", None)
    if deployment_changed:
        config.pop("api_key", None)
    save_config(config)
    return deployment_changed


def set_config_base_url(base_url: str):
    """Deprecated compatibility alias for ``set_config_api_url``."""
    return set_config_api_url(base_url)


def set_human_readable_dates(datetime_columns, df):
    for col in datetime_columns:
        if col in df.columns:
            # Convert UTC string to local time string
            df = df.with_columns(
                pl.col(col)
                .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%Z")
                .map_elements(
                    lambda dt: dt.replace(tzinfo=timezone.utc)
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M:%S %Z")
                    if dt
                    else None,
                    return_dtype=pl.Utf8,
                )
                .alias(col)
            )
    return df


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # Configuration commands must remain available before authentication.
    api_key, api_url, api_url_error = resolve_api_configuration_with_context()
    if (api_key is None or api_url is None) and ctx.invoked_subcommand not in [
        "login",
        "set-api-url",
        "set-base-url",
    ]:
        click.echo(
            api_url_error
            or "Configure SUTRO_API_URL and SUTRO_API_KEY, or run 'sutro login'."
        )
        ctx.exit(1)

    if ctx.invoked_subcommand is None:
        message = """
Welcome to the Sutro CLI! 

To see a list of all available commands, use 'sutro --help'.
    """
        click.echo(Fore.GREEN + message + Style.RESET_ALL)

        click.echo(ctx.get_help())


@cli.command()
def login():
    """Configure a Sutro deployment URL and API key."""
    if API_KEY_ENV in os.environ or API_URL_ENV in os.environ:
        click.echo(
            Fore.YELLOW
            + "Warning: SUTRO_API_URL and SUTRO_API_KEY environment variables "
            + "take precedence over credentials saved by 'sutro login'. Unset "
            + "them to use the saved credentials."
            + Style.RESET_ALL
        )
    default_api_key, default_api_url = resolve_api_configuration()
    default_api_key = default_api_key or ""
    default_api_url = default_api_url or ""

    api_url = click.prompt(
        "Enter your Sutro deployment URL",
        default=default_api_url or None,
        show_default=bool(default_api_url),
    )
    try:
        api_url = normalize_api_url(api_url)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        normalized_default_api_url = (
            normalize_api_url(default_api_url) if default_api_url else None
        )
    except ValueError:
        normalized_default_api_url = None
    if api_url != normalized_default_api_url:
        # Deployment keys are scoped. Never offer a key resolved for one URL
        # as the default after the user chooses another deployment.
        default_api_key = ""

    click.echo(
        "Hint: An API key is already set. Press Enter to keep the existing key."
        if default_api_key
        else ""
    )
    api_key = click.prompt(
        "Enter your API key",
        default=default_api_key,
        hide_input=True,
        show_default=False,
    )

    result = Sutro(api_key=api_key, api_url=api_url).try_authentication(api_key)
    if not result or result.get("authenticated") is not True:
        raise click.ClickException(
            Fore.RED + "Invalid API key. Try again." + Style.RESET_ALL
        )
    else:
        ascii = """

 
 ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌       ▐░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌
▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░▌       ▐░▌
 ▀▀▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌     ▐░▌     ▐░█▀▀▀▀█░█▀▀ ▐░▌       ▐░▌
          ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌     ▐░▌  ▐░▌       ▐░▌
 ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░▌       ▐░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀       ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀ 
                                                                 

"""
        click.echo(Fore.BLUE + ascii + Style.RESET_ALL)
        click.echo(
            Fore.GREEN + "Successfully authenticated. Welcome back!" + Style.RESET_ALL
        )

    config = load_config()
    config.update({"api_key": api_key, "api_url": api_url})
    config.pop("base_url", None)
    save_config(config)


@cli.group()
def jobs():
    """Manage jobs."""
    pass


@jobs.command()
@click.option(
    "--all", is_flag=True, help="Include all jobs, including cancelled and failed ones."
)
def list(all=False):
    """Lists historical and ongoing jobs. Will only list first 25 jobs by default. Use --all to see all jobs."""
    sdk = get_sdk()
    jobs = sdk.list_jobs()
    if jobs is None or len(jobs) == 0:
        click.echo(Fore.YELLOW + "No jobs found." + Style.RESET_ALL)
        return

    df = pl.DataFrame(jobs)
    # TODO: this is a temporary fix to remove jobs where datetime_created is null. We should fix this on the backend.
    df = df.filter(pl.col("datetime_created").is_not_null())
    df = df.sort(by=["datetime_created"], descending=True)

    # Format all datetime columns with a more readable format
    datetime_columns = [
        "datetime_created",
        "datetime_added",
        "datetime_started",
        "datetime_completed",
    ]
    df = set_human_readable_dates(datetime_columns, df)

    # TODO: get colors working
    # df = df.with_columns([
    #     pl.when(pl.col("status") == "SUCCEEDED")
    #     .then(pl.concat_str([pl.lit(Fore.GREEN), pl.col("status"), pl.lit(Style.RESET_ALL)]))
    #     .when(pl.col("status").is_in(["FAILED", "CANCELLED", "UNKNOWN"]))
    #     .then(pl.concat_str([pl.lit(Fore.RED), pl.col("status"), pl.lit(Style.RESET_ALL)]))
    #     .otherwise(pl.col("status"))
    #     .alias("status")
    # ])

    # fill null input_tokens and output_tokens with 0
    df = df.with_columns(
        pl.col("input_tokens").fill_null(0).alias("input_tokens"),
        pl.col("output_tokens").fill_null(0).alias("output_tokens"),
    )

    # fill null datetime_completed with empty string
    df = df.with_columns(
        pl.col("datetime_completed").fill_null("").alias("datetime_completed")
    )

    df = df.with_columns(
        pl.col("job_cost")
        .fill_null(0)
        .map_elements(lambda x: f"${x:.5f}", return_dtype=pl.Utf8)
        .alias("job_cost")
    )

    if all == False:
        df = df.slice(0, 25)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, set_fmt_str_lengths=45):
        print(df.select(pl.all()))


@jobs.command()
@click.argument("job_id")
def status(job_id):
    """Get the status of a job."""
    sdk = get_sdk()
    job_status = sdk.get_job_status(job_id)
    if not job_status:
        return

    print(job_status)


@jobs.command()
@click.argument("job_id")
@click.option(
    "--include-inputs", is_flag=True, help="Include the inputs in the results."
)
@click.option(
    "--include-cumulative-logprobs",
    is_flag=True,
    help="Include the cumulative logprobs in the results.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Download the results to the current working directory. The file name will be the job_id.",
)
@click.option(
    "--save-format",
    type=click.Choice(["parquet", "csv"]),
    default="parquet",
    help="The format of the output file. Options: parquet, csv",
)
def results(
    job_id,
    include_inputs,
    include_cumulative_logprobs,
    save=False,
    save_format="parquet",
):
    """Get the results of a job."""
    sdk = get_sdk()
    job_results = sdk.get_job_results(
        job_id, include_inputs, include_cumulative_logprobs
    )
    if job_results is None or len(job_results) == 0:
        print(Fore.YELLOW + "No results found for job " + job_id + "." + Style.RESET_ALL)
        return

    df = pl.DataFrame(job_results)
    if not save:
        print(df)
    elif save:
        if save_format == "parquet":
            df.write_parquet(f"{job_id}.parquet")
        else:  # csv
            df.write_csv(f"{job_id}.csv")
        print(Fore.GREEN + f"Results saved to {job_id}.{save_format}" + Style.RESET_ALL)


@jobs.command()
@click.argument("job_id")
def cancel(job_id):
    """Cancel a running job."""
    sdk = get_sdk()
    result = sdk.cancel_job(job_id)
    if not result:
        return

    click.echo(Fore.GREEN + "Job cancelled successfully." + Style.RESET_ALL)


@cli.group()
def cache():
    """Manage the local job results cache."""
    pass


@cache.command()
def clear():
    """Clear the local job results cache."""
    sdk = get_sdk()
    sdk._clear_job_results_cache()
    click.echo(Fore.GREEN + "Job results cache cleared." + Style.RESET_ALL)


@cache.command()
def show():
    """Show the contents and size of the job results cache."""
    sdk = get_sdk()
    sdk._show_cache_contents()


@cli.command()
def docs():
    """Open the Sutro API docs."""
    click.launch("https://docs.sutro.sh")


@cli.command("set-api-url")
@click.argument("api_url")
def set_api_url(api_url):
    """Set the Sutro deployment URL for Sutro API requests."""
    try:
        deployment_changed = set_config_api_url(api_url)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        Fore.GREEN + f"API URL set to {normalize_api_url(api_url)}." + Style.RESET_ALL
    )
    if deployment_changed:
        click.echo(
            Fore.YELLOW
            + "The persisted API key was cleared because the deployment changed. "
            + "Run 'sutro login' to configure a key for this deployment."
            + Style.RESET_ALL
        )


@cli.command("set-base-url", hidden=True)
@click.argument("base_url")
def set_base_url(base_url):
    """Deprecated alias for set-api-url."""
    try:
        deployment_changed = set_config_base_url(base_url)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        Fore.YELLOW
        + "set-base-url is deprecated; use set-api-url. "
        + f"API URL set to {normalize_api_url(base_url)}."
        + Style.RESET_ALL
    )
    if deployment_changed:
        click.echo(
            Fore.YELLOW
            + "The persisted API key was cleared because the deployment changed. "
            + "Run 'sutro login' to configure a key for this deployment."
            + Style.RESET_ALL
        )


@cli.command()
def quotas():
    """Get API quotas."""
    sdk = get_sdk()
    quotas = sdk.get_quotas()
    if not quotas:
        return
    print(Fore.YELLOW + "Your current quotas are: \n" + Style.RESET_ALL)
    for priority in range(len(quotas)):
        quota = quotas[priority]
        print(f"Job Priority: {priority}")
        print(f"\tRow Quota (Maximum): {quota['row_quota']}")
        print(f"\tToken Quota (Maximum): {quota['token_quota']}")
        print("\n")
    print(
        Fore.YELLOW
        + "To increase your quotas, contact us at team@sutro.sh."
        + Style.RESET_ALL
    )


@cli.group()
def functions():
    """Run published Functions."""
    pass


def load_json_input(raw: str):
    """Parse a --input value: JSON, or @path to a file holding JSON."""
    source = raw
    if raw.startswith("@"):
        path = raw[1:]
        try:
            with open(path, "r") as handle:
                source = handle.read()
        except OSError as exc:
            raise click.ClickException(f"Could not read {path}: {exc}") from exc
    try:
        return json.loads(source)
    except ValueError as exc:
        subject = (
            f"{raw[1:]} does not contain"
            if raw.startswith("@")
            else "--input is not"
        )
        raise click.ClickException(f"{subject} valid JSON: {exc}") from exc


@functions.command("run")
@click.argument("name")
@click.option(
    "--input",
    "input_value",
    required=True,
    help="Input fields as a JSON object, or @path/to/input.json.",
)
@click.option(
    "--no-confidence-scoring",
    is_flag=True,
    help="Skip the confidence score.",
)
def run(name, input_value, no_confidence_scoring):
    """Run a Function on one input and print the JSON response."""
    payload = load_json_input(input_value)
    sdk = get_sdk()
    try:
        result = sdk.run_function(
            name, payload, confidence_scoring=not no_confidence_scoring
        )
    except requests.HTTPError as exc:
        # The deployment's own explanation is more useful than the status line.
        raise click.ClickException(getattr(exc, "detail", None) or str(exc)) from exc
    click.echo(json.dumps(dict(result), indent=2))


@jobs.command()
@click.argument("job_id", required=False)
@click.option("--latest", is_flag=True, help="Attach to the latest job.")
def attach(job_id, latest):
    """Attach to a running job and stream its progress."""
    sdk = get_sdk()
    if latest:
        jobs = sdk.list_jobs()
        if not jobs:
            click.echo(Fore.YELLOW + "No jobs found." + Style.RESET_ALL)
            return
        job_id = jobs[0]["job_id"]
        print(f"Attaching to latest job: {job_id}")
    elif not job_id:
        click.echo(Fore.YELLOW + "No job ID provided." + Style.RESET_ALL)
        return
    sdk.attach(job_id)


if __name__ == "__main__":
    cli()
