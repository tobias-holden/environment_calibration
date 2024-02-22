from simulations.helpers import load_coordinator_df


def load_sites():
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    #Hi Ricky is removing this for now, but we should develop a new way to remove skipped sites
    sites = coord_df.index.tolist()
    #subsets = coord_df['validation_subset'].tolist()
    nSims = coord_df['nSims'].tolist()
    #script_names = coord_df['run_script_name']
    return sites, nSims#, script_names


if __name__ == '__main__':
    load_sites()
