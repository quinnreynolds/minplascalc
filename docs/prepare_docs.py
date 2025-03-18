import pathlib
import shutil

# Remove the existing API documentation.
shutil.rmtree("_api", ignore_errors=True)
# Create the directory for backreferences.
pathlib.Path("source/backreferences").mkdir(parents=True, exist_ok=True)
