"""Check the ZZ* normalization: sum of mcWeight in the unskimmed file vs sum_of_weights.

Reads only the mcWeight branch over https (network). Writes results/normalization_check.json.
"""

import atlasopenmagic as atom
import numpy as np
import uproot

from higgs4l.data import file_urls, load_config, write_result

DID = 700600


def main():
    cfg = load_config()
    total, entries = 0.0, 0
    for url in file_urls(DID, cfg["release"], "noskim"):
        tree = uproot.open(url, handler=uproot.source.http.HTTPSource)["analysis"]
        total += tree["mcWeight"].array(library="np").astype(np.float64).sum()
        entries += tree.num_entries
        sum_of_weights = float(tree["sum_of_weights"].array(entry_stop=1)[0])
    meta = atom.get_metadata(str(DID))
    result = {
        "did": DID,
        "noskim_entries": entries,
        "metadata_nEvents": meta["nEvents"],
        "noskim_sum_mcWeight": total,
        "sum_of_weights_branch": sum_of_weights,
        "ratio": total / sum_of_weights,
    }
    write_result("normalization_check", result)
    print(result)


if __name__ == "__main__":
    main()
