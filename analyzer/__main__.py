from analyzer.cli import runCli

if __name__ == "__main__":
    args = runCli()
    print("args:", args)
    if hasattr(args, "func"):
        print("ARGS:", args)
        args.func(args)
