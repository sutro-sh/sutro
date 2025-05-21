from datetime import datetime, timezone
import click
from colorama import Fore, Style
import os
import json
from sutro.sdk import Sutro
import polars as pl
import warnings

warnings.filterwarnings("ignore", category=pl.PolarsInefficientMapWarning)
pl.Config.set_tbl_hide_dataframe_shape(True)

CONFIG_DIR = os.path.expanduser("~/.sutro")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_config(config):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def check_auth():
    config = load_config()
    return config.get("api_key") is not None


def get_sdk():
    config = load_config()
    if config.get("base_url") != None:
        return Sutro(
            api_key=config.get("api_key"), base_url=config.get("base_url")
        )
    else:
        return Sutro(api_key=config.get("api_key"))


def set_config_base_url(base_url: str):
    config = load_config()
    config["base_url"] = base_url
    save_config(config)


def set_human_readable_dates(datetime_columns, df):
    for col in datetime_columns:
        if col in df.columns:
            # Convert UTC string to local time string
            df = df.with_columns(
                pl.col(col)
                .str.to_datetime()
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
    # Allow login and set-base-url commands without authentication
    if not check_auth() and ctx.invoked_subcommand not in ["login", "set-base-url"]:
        click.echo("Please login using 'sutro login'.")
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
    """Set or update your API key for Sutro."""
    config = load_config()
    default_api_key = config.get("api_key", "")
    default_base_url = config.get("base_url", "https://api.sutro.sh")
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

    result = get_sdk().try_authentication(api_key)
    if not result or "authenticated" not in result or result["authenticated"] != True:
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

    save_config({"api_key": api_key, "base_url": default_base_url})


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
    if not job_results:
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
def datasets():
    """Manage datasets."""
    pass


@datasets.command()
def create():
    """Create a new dataset."""
    sdk = get_sdk()
    dataset_id = sdk.create_dataset()
    if not dataset_id:
        return
    click.echo(
        Fore.GREEN
        + f"Dataset created successfully. Dataset ID: {dataset_id}"
        + Style.RESET_ALL
    )


@datasets.command()
def list():
    """List all datasets."""
    sdk = get_sdk()
    datasets = sdk.list_datasets()
    if datasets is None or len(datasets) == 0:
        click.echo(Fore.YELLOW + "No datasets found." + Style.RESET_ALL)
        return
    df = pl.DataFrame(datasets)

    df = df.with_columns(
        pl.col("schema")
        .map_elements(lambda x: str(x), return_dtype=pl.Utf8)
        .alias("schema")
    )

    # Format all datetime columns with a more readable format
    datetime_columns = ["datetime_added", "updated_at"]
    df = set_human_readable_dates(datetime_columns, df)

    df = df.sort(by=["datetime_added"], descending=True)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, set_fmt_str_lengths=45):
        print(df.select(pl.all()))


@datasets.command()
@click.argument("dataset_id")
def files(dataset_id):
    """List all files in a dataset."""
    sdk = get_sdk()
    files = sdk.list_dataset_files(dataset_id)
    if not files:
        return

    print(Fore.YELLOW + "Files in dataset " + dataset_id + ":" + Style.RESET_ALL)
    for file in files:
        print(f"\t{file}")


@datasets.command()
@click.argument("dataset_id", required=False)
@click.argument("file_path")
def upload(file_path, dataset_id):
    """Upload files to a dataset. You can provide a single file path or a directory path to upload all files in the directory."""
    sdk = get_sdk()
    sdk.upload_to_dataset(file_path, dataset_id)


@datasets.command()
@click.argument("dataset_id")
@click.argument("file_name", required=False)
@click.argument("output_path", required=False)
def download(dataset_id, file_name=None, output_path=None):
    """Download a file/files from a dataset. If no files are provided, all files in the dataset will be downloaded. If no output path is provided, the file will be saved to the current working directory."""
    sdk = get_sdk()
    files = sdk.download_from_dataset(dataset_id, [file_name], output_path)
    if not files:
        return
    for file in files:
        if output_path is None:
            with open(file_name, "wb") as f:
                f.write(file)
        else:
            with open(output_path + "/" + file_name, "wb") as f:
                f.write(file)


@cli.command()
def docs():
    """Open the Sutro API docs."""
    click.launch("https://docs.sutro.sh")


@cli.command()
@click.argument("base_url")
def set_base_url(base_url):
    """Set the base URL for the Sutro API."""
    set_config_base_url(base_url)
    click.echo(Fore.GREEN + f"Base URL set to {base_url}." + Style.RESET_ALL)


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
