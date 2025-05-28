import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, log_loss
import matplotlib.pyplot as plt
import os

def load_data(pred1, pred2, pred3, pred4, pred5, pred6, actuals):
    return (
        pred1['Prediction'].values,
        pred2['Prediction'].values,
        pred3['Prediction'].values,
        pred4['Prediction'].values,
        pred5['Prediction'].values,
        pred6['Prediction'].values,
        actuals['label'].values
    )

def compute_metrics(predictions, actual_labels):
    f1 = f1_score(actual_labels, predictions)
    accuracy = accuracy_score(actual_labels, predictions)
    loss = log_loss(actual_labels, predictions)
    return {
        'F1': f1,
        'Accuracy': accuracy,
        'Loss': loss
    }

def plot_metrics(models, values, metric_name, save_dir):
    plt.figure(figsize=(10, 6))

    if metric_name == 'Accuracy':
        values = [v * 100 for v in values]  # Convert to percentage
        plt.ylabel(f'{metric_name} (%)')
        plt.ylim(0, 100)
        value_labels = [f"{v:.1f}%" for v in values]
    elif metric_name == 'F1':
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        value_labels = [f"{v:.2f}" for v in values]
    else:  # Loss
        plt.ylabel(metric_name)
        value_labels = [f"{v:.4f}" for v in values]

    bars = plt.bar(models, values)
    plt.title(f'Model Comparison - {metric_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Annotate bars with values
    for bar, label in zip(bars, value_labels):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, label,
                 ha='center', va='bottom', fontsize=10)

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{metric_name.lower()}_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"{metric_name} plot saved to: {save_path}")

def load_rnaernie_data(pred6):
    # Use RNAErnie's own file for both prediction and true label
    pred6_values = pred6['Prediction'].values
    true_labels = pred6['Label'].values
    return pred6_values, true_labels

def main():
    base_path = ''

    pred1 = pd.read_csv('')
    pred2 =  pd.read_csv('')
    pred3 = pd.read_csv('')
    pred4 = pd.read_csv('')
    pred5 = pd.read_csv('')
    pred6 = pd.read_csv('')

    actuals =  pd.read_csv('')

    # Load predictions and labels for RNAErnie from its own file
    preds6, rnaernie_labels = load_rnaernie_data(pred6)

    # Load others as usual
    preds1, preds2, preds3, preds4, preds5, _, true_labels = load_data(
        pred1, pred2, pred3, pred4, pred5, pred6, actuals
    )

    print(f"Length of RNAErnie predictions: {len(preds6)}")
    print(f"Length of RNAErnie labels: {len(rnaernie_labels)}")

    metrics1 = compute_metrics(preds1, true_labels)
    metrics2 = compute_metrics(preds2, true_labels)
    metrics3 = compute_metrics(preds3, true_labels)
    metrics4 = compute_metrics(preds4, true_labels)
    metrics5 = compute_metrics(preds5, true_labels)
    metrics6 = compute_metrics(preds6, rnaernie_labels)

    models = ['DNABERT-1', 'Caduceus', 'Genmask', 'DNABERT-2', 'RNA-FM', 'RNAErnie']
    accuracies = [metrics1['Accuracy'], metrics2['Accuracy'], metrics3['Accuracy'],
                  metrics4['Accuracy'], metrics5['Accuracy'], metrics6['Accuracy']]
    losses = [metrics1['Loss'], metrics2['Loss'], metrics3['Loss'],
              metrics4['Loss'], metrics5['Loss'], metrics6['Loss']]
    f1_scores = [metrics1['F1'], metrics2['F1'], metrics3['F1'],
                 metrics4['F1'], metrics5['F1'], metrics6['F1']]

    plot_dir = os.path.join(base_path, 'plots')
    plot_metrics(models, accuracies, 'Accuracy', plot_dir)
    plot_metrics(models, losses, 'Loss', plot_dir)
    plot_metrics(models, f1_scores, 'F1', plot_dir)

    print("\n=== Final Evaluation Metrics ===")
    for model, acc, loss, f1 in zip(models, accuracies, losses, f1_scores):
        print(f"{model}: Accuracy = {acc*100:.2f}%, Loss = {loss:.4f}, F1 = {f1:.4f}")

    summary_df = pd.DataFrame({
        'Model': models,
        'Accuracy (%)': [round(acc * 100, 2) for acc in accuracies],
        'Loss': [round(l, 4) for l in losses],
        'F1 Score': [round(f1, 4) for f1 in f1_scores]
    })

    summary_csv_path = os.path.join(plot_dir, 'model_metrics_summary_with_earlystop.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📄 Metrics summary saved to: {summary_csv_path}")



if __name__ == "__main__":
    main()

   
