# core/ML/ModelSummary.py
from sklearn.model_selection import train_test_split
from avstats.core.ML.ModelEvaluation import *
from avstats.core.ML.ModelTraining import ModelTraining


def modelling_summary(dataframes, titles, models_to_train, output_file="model_summaries.csv"):
    """
    Train and evaluate multiple models on different datasets.

    Parameters:
    - dataframes: List of tuples [(DataFrame, str)], each containing the dataset and target column name.
    - titles: List of titles corresponding to the datasets.
    - models_to_train: List of model names as keys from the `ModelTraining.models` dictionary.
    - output_file: File path to save the model summaries as a CSV file.

    Returns:
    - dict: Dictionary with dataset titles as keys and their trained models and metrics as values.
    - pd.DataFrame: DataFrame containing the overall model summaries.
    - dict: Simplified dictionary with only the key metrics (MAE, MAPE, RMSE) for each model and dataset.
    """
    num_plots = len(dataframes) * len(models_to_train)
    rows = (num_plots + 1) // 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    axes = axes.flatten()

    model_summaries = []
    plot_idx = 0
    metrics_summary = {}

    for i, ((df, column), title) in enumerate(zip(dataframes, titles)):
        try:
            # Prepare features and target
            x = df.drop(columns=[column])
            y = df[column]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model_training = ModelTraining(x_train, y_train, x_test, y_test)

            dataset_metrics = {}

            for model_name in models_to_train:
                # Train the model
                model_function = model_training.models[model_name]
                trained_model, predictions = model_function()

                # Evaluate the model
                residuals = y_test - predictions
                metrics = evaluate_model(y_test, predictions, residuals)

                # Store simplified metrics
                dataset_metrics[model_name] = {
                    "MAE (min.)": metrics["MAE (min.)"],
                    "MAPE (%)": metrics["MAPE (%)"],
                    "RMSE (min.)": metrics["RMSE (min.)"],
                }

                # Plot results
                ax = axes[plot_idx]
                model_training.plot_model(title=f"{model_name} - {title}", evaluation_metrics=metrics, ax=ax)
                plot_idx += 1

                # Cross-validation
                cv_scores = cross_validate(x_train, y_train)

                # Store results for summaries
                model_summary = {
                    "Dataset": title,
                    "Model": model_name,
                    "MAE (min.)": metrics["MAE (min.)"],
                    "MAPE (%)": metrics["MAPE (%)"],
                    "RMSE (min.)": metrics["RMSE (min.)"],
                    "Mean CV R2": cv_scores.mean(),
                    "Std CV R2": cv_scores.std(),
                    "CV Scores": cv_scores.tolist(),
                }
                model_summaries.append(model_summary)

            metrics_summary[title] = dataset_metrics

        except KeyError as e:
            print(f"Skipping DataFrame {i} due to missing column: {e}")

    # Hide unused axes
    for ax in axes[plot_idx:]:
        ax.set_visible(False)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

    # Convert model summaries to a DataFrame and save
    model_summary_df = pd.DataFrame(model_summaries)
    model_summary_df.to_csv(output_file, index=False)

    return model_summary_df, metrics_summary
