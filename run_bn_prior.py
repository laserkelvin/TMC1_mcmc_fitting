import repack


PROGRESSBAR = True

input_dict = repack.load_input_file("yml/benzonitrile.yml")

output_npy, catalog, output_path = repack.init_setup(**input_dict)

# run the simulation
repack.fit_multi_gaussian(
    output_npy,
    output_path=output_path,
    catalogue=catalog,
    progressbar=PROGRESSBAR,
    **input_dict
)
