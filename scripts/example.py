import time
import argparse
import os

from hyperparameters import sample, metaheuristic, combine, score, stats

if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Directory with wordlist and translate file")
    parser.add_argument("-p", "--profile", type=str, required=False, default="", help="Parameter profile to use")
    args = parser.parse_args()

    datadir = args.datadir.rstrip("/")
    wl_file, tr_file = "", ""
    for file in os.listdir(datadir):
        if file.endswith(".wlh"):
            wl_file = datadir + "/" + file
        elif file.endswith(".tra"):
            tr_file = datadir + "/" + file

    if not wl_file or not tr_file:
        print(f"Wordlist or translate file not present in {datadir} directory")
        exit(1)

    scorer = score.PatgenScorer(
        "patgen", wl_file, tr_file, verbose=True
    )

    if not args.profile:
        par_file = ""
        par_dir = datadir
        for _ in range(3):  # assume the directory structure .../data/<lang>/<dataset>
            if "patgen_params.in" in os.listdir(par_dir):
                par_file = par_dir + "/patgen_params.in"
                break
            par_dir = par_dir + "/.."

        if not par_file:
            print(f"Patgen parameters file <patgen_params.in> not found in {datadir} or 2 level above")
            exit(1)
    else:
        par_file = args.profile

    sampler = sample.FileSampler(par_file)
    statistic = stats.LearningInfo()

    meta = metaheuristic.NoMetaheuristic(
        scorer,
        sampler,
        statistic=statistic
    )

    comb = combine.SimpleCombiner(meta, verbose=True)

    comb.run()
    #print([(pop.f_score(1.0), pop.f_score(100.0)) for pop in meta.population])
    meta.statistic.visualise(metric=["precision", "recall"])
    print([(l[0].stats["tp"], l[0].stats["fp"], l[0].stats["fn"], l[0].stats["level_patterns"]) for l in meta.statistic.level_outputs])

    #for s in meta.population:
    #    print(str(stats.PatternsInfo(f"{datadir}/{s.timestamp}-{s.run_id}.pat", s)))

    print("Ran", round(time.time() - t, 2), "seconds")
