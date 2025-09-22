import os
from gp_pipeline.models.base import GPModelPipeline
from gp_pipeline.utils.plotting import plot_losses, plotGPTrueDifference, plotEntropyHistogram, plotSlice2D, plotSlice4D
from gp_pipeline.utils.evaluation import evaluate_and_log, misclassified, evaluate_mlp_and_log
from gp_pipeline.utils.physics_interface import Run3PhysicsInterface
from gp_pipeline.utils.create_config import create_config

USER = os.environ.get("USER")

def make_gp_pipeline(cfg):
    return GPModelPipeline(config=cfg)

def generate_name(cfg):
    '''Function to generate names for gof_csv, checkpoints and training data'''
    base = f"{cfg.target}_{cfg.n_dim}D"
    additional_parts = []
    for param in getattr(cfg, "__sweep_params__", []):
        value = getattr(cfg, param, "NA")
        additional_parts.append(f"{param}_{value}")
    additional = "_" + "_".join(additional_parts) if additional_parts else ""
    folder_name = base + additional
    if cfg.is_deep:
        base += "_deep_"
    if cfg.is_sparse:
        base += "_sparse_"
    if cfg.is_mlp:
        base += "_mlp_"
    if cfg.is_active:
        base += "_AL"
    else:
        if cfg.is_lh:
            base += "_LH_"
        else:
            base += "_random_"
    base += f"_run{cfg.run}"

    # If MLP with AL points
    if cfg.train_mlp_with_previous_al_points:
        base += "mlp_with_al_points"

    print(f"[INFO] Base name: {base}")
    print(f"[INFO] Additional name: {additional}")

    # Training data path
    total_name = base + additional
    return f"{base}", f"{folder_name}", f"{total_name}"

def run_pipeline_iteration(cfg):
    '''Function to run the whole pipeline with or without Active Learning'''    

    print(f"[INFO] Starting ITERATION: {cfg.iteration}")
    print(f"[INFO] run: {cfg.run}")
    print(f"[INFO] active: {cfg.is_active}")

    base_dir = f"/u/{USER}/al_pmssmwithgp/model"

    # Create names for gof_csv, checkpoints and training data
    name, folder_name, total_name = generate_name(cfg)
    eval_base_dir = os.path.join(base_dir+ f'/goodness_of_fit/{cfg.n_dim}D/{folder_name}')
    if not os.path.exists(eval_base_dir):
        os.makedirs(eval_base_dir, exist_ok=True)
    cfg.total_name = total_name


    gp_pipeline = make_gp_pipeline(cfg)
    
    print(f"[INFO] Starting Iteration {cfg.iteration} with Active Learning == {cfg.is_active}")

    # Load training data (also for combined approach: first run select AL points and then train MLP on these points)
    if cfg.prepare_for_mlp and cfg.is_active and not any([cfg.is_deep, cfg.is_mlp, cfg.is_sparse]):
        training_data_path_mlp = os.path.join(base_dir+ f"/training_data", f'training_data_{cfg.target}{cfg.n_dim}D_AL_run{cfg.run}_iteration{cfg.iteration}.pkl')
    if cfg.train_mlp_with_previous_al_points:
        training_data_path = os.path.join(base_dir+ f"/training_data", f'training_data_{cfg.target}{cfg.n_dim}D_AL_run{cfg.run}_iteration{cfg.iteration-1}.pkl')
    else:
        training_data_path = os.path.join(base_dir+ f"/training_data", f'training_data_{total_name}.pkl')

    if cfg.iteration > 1 or cfg.evaluation_mode:
        if os.path.exists(training_data_path):
            gp_pipeline.load_training_data(training_data_path)
            print(f"[INFO] Training data successfully loaded from {training_data_path}")

    # Initialize model
    gp_pipeline.initialize_model()
    print("[INFO] Model was successfully initialized")

    # Warm starting - use weights from previous trainings
    saved_model_path = os.path.join(base_dir+ f"/checkpoints", f'model_checkpoint_{total_name}.pth')
    if cfg.warm_starting and not cfg.is_mlp:
        if cfg.iteration > 1: # and cfg.is_active: 
            try: 
                gp_pipeline.load_model(saved_model_path)
                print(f"[INFO] Warm Start with saved model parameters from {saved_model_path}")
                #print("[INFO] Loaded feature weights sum:", gp_pipeline.model.feature_extractor.net[0].weight.sum().item())
            except RuntimeError as e:
                print(f"[WARNING] Could not load model parameters {e}")
        
    # If only evaluation just load model, otherwise train model
    if cfg.evaluation_mode:
        print(f"[INFO] Evaluate Model from {saved_model_path}")
        gp_pipeline.load_model(saved_model_path)
    else:
        print("[INFO] Start Training")
        gp_pipeline.train_model()
        if cfg.is_mlp_with_al:
            print("[INFO] Start Training (MLP parallel)")
            gp_pipeline.train_mlp_parallel()

        # Plot loss curves
        # if not cfg.is_mlp:
        loss_base_dir = os.path.join(base_dir+ f"/loss_plots")
        if not os.path.exists(loss_base_dir):
            os.makedirs(loss_base_dir, exist_ok=True)
        loss_plot_path = os.path.join(loss_base_dir, f'loss_plot_{total_name}.png')
        plot_losses(gp_pipeline.losses, gp_pipeline.losses_valid, loss_plot_path, cfg.iteration)
    
        # Save lengthscales 
        if not cfg.is_deep and not cfg.is_mlp:
            lengthscale_base_dir = os.path.join(base_dir+ f"/lengthscales")
            if not os.path.exists(lengthscale_base_dir):
                os.makedirs(lengthscale_base_dir, exist_ok=True)
            lengthscale_path = os.path.join(lengthscale_base_dir, f'lengthscales_{total_name}.csv')
            gp_pipeline.save_lengthscale(lengthscale_path)

    # Evaluate the model
    if cfg.evaluation_mode:
        gp_pipeline.evaluate_model()
    
    # Calculate accuracy
    if not cfg.evaluation_mode:
        print("[INFO] Caclculate metrics")
        evaluate_and_log(gp_pipeline, cfg, name, eval_base_dir)
        if cfg.is_mlp_with_al:
            evaluate_mlp_and_log(gp_pipeline, cfg, name, eval_base_dir)

    # Plot prediction
    if not cfg.is_mlp:
        plotGPTrueDifference(gp_pipeline, 
                                save_path=os.path.join(base_dir+ f"/plots", f'plotGPTrueDifference_{total_name}.png'), 
                                iteration=cfg.iteration)
        # plotEntropyHistogram(gp_pipeline, 
        #                 save_path=os.path.join(base_dir+ f"/plots", f'plotEntropyHistogram{total_name}.png'), 
        #                 iteration=cfg.iteration)

    # Sample new points either with Active Learning or randomly
    #if cfg.is_active:
    if not cfg.train_mlp_with_previous_al_points:
        new_points = gp_pipeline.select_new_points(N=cfg.n_new_points)
        print(f"[INFO] New points selected: {new_points}")
        # Generate new points with Run3ModelGen or for Toy model
        if cfg.target in ["DMRD", "CrossSection", "CLs"]:
            new_points_unnormalized = gp_pipeline._unnormalize(new_points)
            print(f"[INFO] New points unnormalized: {new_points_unnormalized}")
            create_config(new_points=new_points_unnormalized, n_dim=cfg.n_dim, output_file =f'new_config_{total_name}.yaml')
            physics_interface = Run3PhysicsInterface()
            new_root_file = physics_interface.generate_targets(cfg.iteration, cfg.n_dim, total_name, target=cfg.target, seed=cfg.run)
            print(f"[INFO] New ROOT file generated: {new_root_file}")
            gp_pipeline.load_additional_data(new_points, new_root_file)
        elif cfg.target == "Toy":
            gp_pipeline.load_additional_data(new_points)

    # Create slha files for misclassified points
    if cfg.target == "CLs":
        misclassified_points_unnormalized = gp_pipeline._unnormalize(gp_pipeline.misclassified_points)
        misclassified_name = total_name + '_misclassified'
        print(f"[INFO] Misclassified points unnormalized: {misclassified_points_unnormalized}")
        create_config(new_points=misclassified_points_unnormalized, n_dim=cfg.n_dim, output_file =f'new_config_{misclassified_name}.yaml')
        physics_interface = Run3PhysicsInterface()
        new_root_file = physics_interface.generate_targets(cfg.iteration, cfg.n_dim, misclassified_name, target=cfg.target, seed=cfg.run)
        print(f"[INFO] New ROOT file generated: {new_root_file}")

    # else:
    #     #if not cfg.is_mlp:
    #     # Load the number of points for the next iteration
    #     gp_pipeline.initial_train_points = cfg.initial_train_points + cfg.iteration * (cfg.n_new_points*10) # for 19D creation of RUN3ModelGen 10 times slower than AL
    #     print("[INFO] Number of points points for next iteration: ", gp_pipeline.initial_train_points)
    #     gp_pipeline.load_initial_data(not_active=True)

    # Plot distributions with new points and entropies
    if cfg.n_dim == 4:
        plotSlice4D(gp_pipeline, save_path=os.path.join(base_dir+ f"/plots", f'plotSlice4D_{total_name}.png'), 
                                func="mean", iteration=cfg.iteration, new_x=new_points)
        plotSlice4D(gp_pipeline, save_path=os.path.join(base_dir+ f"/plots", f'plotSlice4DEntropy_{total_name}.png'), 
                                func="entropy", iteration=cfg.iteration)
    elif cfg.n_dim == 2:
        plotSlice2D(gp_pipeline, save_path=os.path.join(base_dir+ f"/plots", f'plotSlice2D_{total_name}.png'), 
                                func="mean", iteration=cfg.iteration, new_x=new_points)
        plotSlice2D(gp_pipeline, save_path=os.path.join(base_dir+ f"/plots", f'plotSlice2DEntropy_{total_name}.png'),
                                func="entropy", iteration=cfg.iteration)

    # Save training data
    if not cfg.train_mlp_with_previous_al_points:
        gp_pipeline.save_training_data(training_data_path)
        print(f"[INFO] Training data saved to: {training_data_path}")

    if cfg.prepare_for_mlp:
        gp_pipeline.save_training_data(training_data_path_mlp)
        print(f"[INFO] Training data saved for later training with MLP to: {training_data_path_mlp}")

    # Save model if not MLP 
    if not cfg.is_mlp:
        gp_pipeline.save_model(saved_model_path)

    print(f"[INFO] Iteration {cfg.iteration} completed.")