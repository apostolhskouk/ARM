import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Optional

class EvaluationMetrics:
    """
    Calculates and visualizes retrieval evaluation metrics (P, R, F1, PR) at various @n values.
    Includes a column indicating the percentage of instances with fewer unique predictions than n.
    (Code from previous step)
    """
    def __init__(self, n_values: List[int]):
        if not n_values:
            raise ValueError("n_values list cannot be empty.")
        self.n_values = sorted(list(set(n_values))) # Ensure unique and sorted
        self.results_df = None # Placeholder for a single result set

    def _get_unique_ordered(self, ids: List[str]) -> List[str]:
        """Returns a list of unique IDs preserving original order."""
        seen: Set[str] = set()
        unique_ids: List[str] = []
        for item_id in ids:
            if item_id not in seen:
                unique_ids.append(item_id)
                seen.add(item_id)
        return unique_ids

    def _calculate_single_instance_metrics_at_n(
        self,
        unique_gt_ids: Set[str],
        unique_ordered_pred_ids: List[str],
        n: int
    ) -> Dict[str, float]:
        """Calculates metrics for a single instance at a specific n."""
        top_n_pred_ids_list = unique_ordered_pred_ids[:n]
        top_n_pred_ids_set = set(top_n_pred_ids_list)

        true_positives: int = len(unique_gt_ids.intersection(top_n_pred_ids_set))
        num_pred: int = len(top_n_pred_ids_list)
        num_gt: int = len(unique_gt_ids)

        precision: float = true_positives / num_pred if num_pred > 0 else 0.0
        recall: float = 1.0 if num_gt == 0 else true_positives / num_gt
        f1: float = (
            0.0
            if (precision + recall) == 0
            else 2 * (precision * recall) / (precision + recall)
        )
        perfect_recall: float = (
             1.0 if num_gt == 0 else float(unique_gt_ids.issubset(top_n_pred_ids_set))
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "perfect_recall": perfect_recall,
        }

    def calculate_metrics(
        self,
        ground_truth_ids: List[List[str]],
        predicted_ids: List[List[str]],
    ) -> pd.DataFrame:
        """
        Calculates average P, R, F1, PR across all instances for each n.
        Also calculates the percentage of instances where unique predictions < n.
        """
        if len(ground_truth_ids) != len(predicted_ids):
            raise ValueError(f"Ground truth ({len(ground_truth_ids)}) and predicted ({len(predicted_ids)}) lists must have the same length.")

        num_instances = len(ground_truth_ids)
        if num_instances == 0:
            print("Warning: Input lists are empty. Returning empty DataFrame.")
            columns = ['@n', 'Precision', 'Recall', 'F1', 'Perfect Recall', 'Perc_Preds_Less_Than_n']
            return pd.DataFrame(columns=columns).set_index('@n')

        metrics_data: Dict[int, List[Dict[str, float]]] = {n: [] for n in self.n_values}
        less_than_n_counts: Dict[int, int] = {n: 0 for n in self.n_values}

        for i in range(num_instances):
            gt_ids_instance = ground_truth_ids[i]
            pred_ids_instance = predicted_ids[i]

            unique_gt_set: Set[str] = set(gt_ids_instance)
            unique_ordered_pred: List[str] = self._get_unique_ordered(pred_ids_instance)
            num_unique_pred = len(unique_ordered_pred)

            for n in self.n_values:
                if num_unique_pred < n:
                    less_than_n_counts[n] += 1
                instance_metrics = self._calculate_single_instance_metrics_at_n(
                    unique_gt_set, unique_ordered_pred, n
                )
                metrics_data[n].append(instance_metrics)

        summary_results: Dict[str, List] = {
            '@n': [], 'Precision': [], 'Recall': [], 'F1': [],
            'Perfect Recall': [], 'Perc_Preds_Less_Than_n': [],
        }

        for n in self.n_values:
            avg_precision = sum(m['precision'] for m in metrics_data[n]) / num_instances
            avg_recall = sum(m['recall'] for m in metrics_data[n]) / num_instances
            avg_f1 = sum(m['f1'] for m in metrics_data[n]) / num_instances
            avg_perfect_recall = sum(m['perfect_recall'] for m in metrics_data[n]) / num_instances
            perc_less_than_n = (less_than_n_counts[n] / num_instances) * 100.0

            summary_results['@n'].append(n)
            summary_results['Precision'].append(avg_precision)
            summary_results['Recall'].append(avg_recall)
            summary_results['F1'].append(avg_f1)
            summary_results['Perfect Recall'].append(avg_perfect_recall)
            summary_results['Perc_Preds_Less_Than_n'].append(perc_less_than_n)

        self.results_df = pd.DataFrame(summary_results).set_index('@n')
        return self.results_df

    def visualize_results(self, df_to_plot: Optional[pd.DataFrame] = None, title: str = "Retrieval Performance Metrics"):
        """
        Plots the performance metrics (P, R, F1, PR) from a given DataFrame.
        """
        plot_df = df_to_plot if df_to_plot is not None else self.results_df

        if plot_df is None:
            print("Error: No results DataFrame provided or stored to visualize.")
            return
        if plot_df.empty:
            print("Warning: Results DataFrame is empty. Nothing to plot.")
            return

        plot_columns = ['Precision', 'Recall', 'F1', 'Perfect Recall']
        plot_df_metrics = plot_df[[col for col in plot_columns if col in plot_df.columns]]

        if plot_df_metrics.empty:
             print("Warning: No standard performance metric columns found in DataFrame. Nothing to plot.")
             return

        ax = plot_df_metrics.plot(kind='line', marker='o', figsize=(10, 6))
        ax.set_xlabel("@n")
        ax.set_ylabel("Score")
        ax.set_title(title)
        # Use the index of the DataFrame being plotted for x-ticks
        if isinstance(plot_df_metrics.index, pd.MultiIndex):
             ax.set_xticks(range(len(plot_df_metrics.index)))
             ax.set_xticklabels(plot_df_metrics.index.values)
        else:
             ax.set_xticks(plot_df_metrics.index)

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Metrics")
        plt.tight_layout()
        plt.show()
    def calculate_metrics_for_single_n(
        self,
        ground_truth_ids: List[List[str]],
        predicted_ids: List[List[str]],
        n_value: int
    ) -> Dict[str, float]:
        if len(ground_truth_ids) != len(predicted_ids):
            raise ValueError("Ground truth and predicted lists must have the same length.")

        num_instances = len(ground_truth_ids)
        if num_instances == 0:
            return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Perfect Recall": 0.0}

        instance_metrics_list: List[Dict[str, float]] = []

        for i in range(num_instances):
            gt_ids_instance = ground_truth_ids[i]
            pred_ids_instance = predicted_ids[i]

            unique_gt_set: Set[str] = set(gt_ids_instance)
            unique_ordered_pred: List[str] = self._get_unique_ordered(pred_ids_instance)

            instance_metrics = self._calculate_single_instance_metrics_at_n(
                unique_gt_set, unique_ordered_pred, n_value
            )
            instance_metrics_list.append(instance_metrics)

        avg_precision = sum(m['precision'] for m in instance_metrics_list) / num_instances if num_instances > 0 else 0.0
        avg_recall = sum(m['recall'] for m in instance_metrics_list) / num_instances if num_instances > 0 else 0.0
        avg_f1 = sum(m['f1'] for m in instance_metrics_list) / num_instances if num_instances > 0 else 0.0
        avg_perfect_recall = sum(m['perfect_recall'] for m in instance_metrics_list) / num_instances if num_instances > 0 else 0.0

        return {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1": avg_f1,
            "Perfect Recall": avg_perfect_recall,
        }
# --- End of Base Class ---
