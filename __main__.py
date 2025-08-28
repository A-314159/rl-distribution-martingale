import utilities.tensorflow_config as cfg

if __name__ == "__main__":

    cfg.configure()

    print("----------------------------------------------------------------------------")
    print("Training will begin.")
    print("In python console, enter: q to save and quit training, c to show distribution chart")
    print("----------------------------------------------------------------------------")

    # note: The import below must be done after call to configure()
    from core.main import main

    main()
