try:
    import setGPU
except:
    print("[WARNING] Unable to import setGPU, hopefully you don't need GPU power.")

__all__ = ["compare_helper", "train_helper", "model_helper", "print_helper", "data_helper"]
