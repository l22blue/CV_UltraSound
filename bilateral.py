from experiment_common import parse_common_args, run_experiment

if __name__ == "__main__":
    args = parse_common_args()
    run_experiment("bilateral", args.root, args.epochs, args.batch_size, args.seed, args.lr, args.num_workers)
