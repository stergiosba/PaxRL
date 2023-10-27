import tomli

def read_config(file_mame):
    if ".toml" in file_mame:
        with open(f"{file_mame}", mode="rb") as config_file:
            config = tomli.load(config_file)
    else:
        with open(f"{file_mame}.toml", mode="rb") as config_file:
            config = tomli.load(config_file)

    return config