import subprocess
import os 
from termcolor import colored


def check_poetry_setup(test_username: str) -> None:
    """
    Testing if everything is setup correctly and json interface works well.
    Parameters: 
    test_username: str - Testo parameter from smoketest.json
    Returns:
    None
    """
    project_directory:str = os.path.dirname(os.path.abspath(__file__))
    toml_file_path:str = os.path.join(project_directory, '..', '..')

    try:
        os.chdir(toml_file_path)
        subprocess.run(["poetry", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["poetry", "check"], check=True)
        print(colored(f"{test_username}, you are ready to go! :)", 'green'))

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(colored(f"Failed to set up Poetry project.", 'red'))

