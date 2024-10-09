import click
from colorama import Fore, Style
import os 
import json
from materialized_intelligence.sdk import MaterializedIntelligence
import polars as pl

CONFIG_DIR = os.path.expanduser("~/.materialized_intelligence")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def check_auth():
    config = load_config()
    return config.get("api_key") is not None

def get_sdk():
    config = load_config()
    if config.get("base_url") != None:
        return MaterializedIntelligence(api_key=config.get("api_key"), base_url=config.get("base_url"))
    else:
        return MaterializedIntelligence(api_key=config.get("api_key"))

def set_config_base_url(base_url: str):
    config = load_config()
    config["base_url"] = base_url
    save_config(config)

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if not check_auth() and ctx.invoked_subcommand != "login":
        click.echo("Please login using 'mi login'.")
        ctx.exit(1)

    if ctx.invoked_subcommand is None:
        message = """
Welcome to the Materialized Intelligence CLI! 

To get started, login with 'mi login'.

To see a list of all available commands, use 'mi --help'.
    """
        click.echo(Fore.GREEN + message + Style.RESET_ALL)
        
        click.echo(ctx.get_help())

@cli.command()
def login():
    """Login to the Materialized Intelligence API."""
    config = load_config()
    default_api_key = config.get("api_key", "")
    click.echo("Hint: An API key is already set. Press Enter to keep the existing key." if default_api_key else "")
    api_key = click.prompt("Enter your API key", default=default_api_key, hide_input=True, show_default=False)

    try_authentication = get_sdk().try_authentication(api_key)
    if 'authenticated' not in try_authentication or try_authentication['authenticated'] != True:
        raise click.ClickException(Fore.RED + "Invalid API key. Try again." + Style.RESET_ALL)
    else:
        ascii = """
    __  ___      __            _       ___               __   
   /  |/  /___ _/ /____  _____(_)___ _/ (_)___ ___  ____/ /   
  / /|_/ / __ `/ __/ _ \/ ___/ / __ `/ / /_  // _ \/ __  /    
 / /  / / /_/ / /_/  __/ /  / / /_/ / / / / //  __/ /_/ /     
/_/ ____\__,_/___/\___/______/\__,_/_/_/ /___|___/\__,_/      
   /  _/___  / /____  / / (_)___ ____  ____  ________         
   / // __ \/ __/ _ \/ / / / __ `/ _ \/ __ \/ ___/ _ \        
 _/ // / / / /_/  __/ / / / /_/ /  __/ / / / /__/  __/        
/___/_/ /_/\__/\___/_/_/_/\__, /\___/_/ /_/\___/\___/         
                         /____/                               
"""
        click.echo(Fore.BLUE + ascii + Style.RESET_ALL)
        click.echo(Fore.GREEN + "Successfully authenticated. Welcome back!" + Style.RESET_ALL)

    save_config({"api_key": api_key})

@cli.command()
def jobs():
    """List all historical jobs."""
    sdk = get_sdk()
    jobs = sdk.list_jobs()
    df = pl.DataFrame(jobs)
    df = df.sort(by=["datetime_created"], descending=True)

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
        pl.col("output_tokens").fill_null(0).alias("output_tokens")
    )

    # fill null datetime_completed with empty string
    df = df.with_columns(
        pl.col("datetime_completed").fill_null("").alias("datetime_completed")
    )

    df = df.with_columns(
        pl.col("job_cost").fill_null(0).map_elements(lambda x: f"${x:.5f}").alias("job_cost")
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, set_fmt_str_lengths=45):
        print(df.select(pl.all()))

@cli.command()
@click.argument("job_id")
def status(job_id):
    """Get the status of a job."""
    sdk = get_sdk()
    job_status = sdk.get_job_status(job_id)['job_status'][job_id]
    print(job_status)

@cli.command()
@click.argument("job_id")
@click.option("--include-inputs", is_flag=True, help="Include the inputs in the results.")
def results(job_id, include_inputs):
    """Get the results of a job."""
    sdk = get_sdk()
    job_results = sdk.get_job_results(job_id, include_inputs)
    if job_results is not None:
        df = pl.DataFrame(job_results)
        print(df)

@cli.command()
@click.argument("job_id")
def cancel(job_id):
    """Cancel a running job."""
    sdk = get_sdk()
    sdk.cancel_job(job_id)
    click.echo(Fore.GREEN + "Job cancelled successfully." + Style.RESET_ALL)

@cli.command()
def docs():
    """Open the Materialized Intelligence API docs."""
    click.launch("https://docs.materialized.dev")

@cli.command()
@click.argument("base_url")
def set_base_url(base_url):
    """Set the base URL for the Materialized Intelligence API."""
    set_config_base_url(base_url)
    click.echo(Fore.GREEN + f"Base URL set to {base_url}." + Style.RESET_ALL)

if __name__ == "__main__":
    cli()