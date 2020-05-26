import os
import getpass


username = getpass.getuser()
DRUG_ROOT_DIR = os.path.join("/home", username, "DEEP_MODEL_temp")
