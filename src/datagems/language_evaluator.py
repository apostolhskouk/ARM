import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
import logging
from collections import defaultdict
import wandb
from src.utils.evaluator import EvaluationMetrics
from src.retrieval.base import RetrievalResult
try:
    from IPython.display import display, HTML
    _IPYTHON_DISPLAY_AVAILABLE = True
except ImportError:
    _IPYTHON_DISPLAY_AVAILABLE = False
    # Define dummy functions if needed, or just rely on the flag
    display = print # A simple fallback if needed outside the main logic
    HTML = lambda x: print(x) # Simple fallback
    
class LanguageEvaluator(EvaluationMetrics):
    """
    Evaluates retriever performance against a benchmark JSONL file,
    calculating metrics for different language scenarios and overall.
    Optionally logs results to Weights & Biases.
    """

    def __init__(self, n_values: List[int]):
        """
        Initializes the evaluator.

        Args:
            n_values: A list of integers representing the 'n' values for
                      calculating metrics@n (e.g., [1, 5, 10]).
        """
        super().__init__(n_values)
        self.scenarios: List[Tuple[str, str]] = [
            ('en', 'en'),
            ('de', 'de'),
            ('fr', 'fr'),
            ('en', 'de'),
            ('en', 'fr'),
        ]
        self.scenario_names: List[str] = [f"{ql}->{dl}" for ql, dl in self.scenarios]

    def _load_and_prepare_data(
        self,
        benchmark_json_path: str,
        retrieved_results: List[List[RetrievalResult]]
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[str]]]]:
        """
        Loads benchmark data and aligns it with retrieved results by scenario.

        Returns:
            A tuple containing two dictionaries:
            1. Ground truth IDs grouped by scenario ("ql->dl" or "Overall").
            2. Predicted IDs grouped by scenario ("ql->dl" or "Overall").
        """
        gt_by_scenario: Dict[str, List[List[str]]] = defaultdict(list)
        pred_by_scenario: Dict[str, List[List[str]]] = defaultdict(list)
        gt_by_scenario["Overall"] = []
        pred_by_scenario["Overall"] = []

        benchmark_data = []
        try:
            with open(benchmark_json_path, 'r', encoding='utf-8') as f:
                try:
                    benchmark_data = json.load(f)
                    if not isinstance(benchmark_data, list):
                        raise TypeError(f"Expected a JSON list in {benchmark_json_path}, but got {type(benchmark_data)}")
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON format in {benchmark_json_path}: {e}")
                    raise
        except FileNotFoundError:
            logging.error(f"Benchmark file not found: {benchmark_json_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading benchmark file {benchmark_json_path}: {e}")
            raise

        if len(benchmark_data) != len(retrieved_results):
            raise ValueError(
                f"Mismatch between number of benchmark queries ({len(benchmark_data)}) "
                f"and number of retrieved result sets ({len(retrieved_results)})."
            )

        for i, benchmark_item in enumerate(benchmark_data):
            query_lang = benchmark_item.get("question_language")
            doc_lang = benchmark_item.get("language")
            source_docs_str = benchmark_item.get("source_document", "")

            if not query_lang or not doc_lang:
                logging.warning(f"Skipping benchmark item {i+1} due to missing language fields.")
                continue

            # Parse ground truth IDs
            gt_ids = [doc_id.strip() for doc_id in source_docs_str.split(',') if doc_id.strip()]

            # Extract predicted IDs
            pred_instance_results = retrieved_results[i]
            pred_ids = [result.metadata.get('id') for result in pred_instance_results if result.metadata.get('id')]
            if len(pred_ids) != len(pred_instance_results):
                 logging.warning(f"Query {i+1}: Some retrieved results missing 'id' in metadata.")


            scenario_key = f"{query_lang}->{doc_lang}"

            # Store data for the specific scenario if it's one we track
            if scenario_key in self.scenario_names:
                gt_by_scenario[scenario_key].append(gt_ids)
                pred_by_scenario[scenario_key].append(pred_ids)

            # Always store for the overall calculation
            gt_by_scenario["Overall"].append(gt_ids)
            pred_by_scenario["Overall"].append(pred_ids)

        return gt_by_scenario, pred_by_scenario

    def evaluate(
        self,
        benchmark_json_path: str,
        retrieved_results: List[List[RetrievalResult]],
        total_seconds: float,
        enable_wandb: bool = False,
        project_wandb: Optional[str] = None,
        entity_wandb: Optional[str] = None,
        group_wandb: Optional[str] = None,
        name_wandb: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Performs the evaluation across defined scenarios and overall.

        Args:
            benchmark_json_path: Path to the JSON benchmark file (containing a list of query objects).
            retrieved_results: The list of lists of RetrievalResult from the retriever.
            total_seconds: Total time taken for the retrieval experiment.
            enable_wandb: If True, log results to Weights & Biases.
            project_wandb: W&B project name (required if enable_wandb=True).
            entity_wandb: W&B entity (user or team) name (required if enable_wandb=True).
            group_wandb: W&B group name (optional).
            name_wandb: W&B run name (optional).

        Returns:
            A dictionary where keys are scenario names (e.g., "en->en", "Overall")
            and values are Pandas DataFrames containing the evaluation metrics for that scenario.
        """
        if enable_wandb:
            try:
                wandb.init(
                    project=project_wandb,
                    entity=entity_wandb,
                    group=group_wandb,
                    name=name_wandb,
                    config={"n_values": self.n_values},
                    job_type="evaluation"
                )
            except Exception as e:
                logging.error(f"Failed to initialize W&B: {e}")
                enable_wandb = False # Disable W&B logging if init fails

        results_tables: Dict[str, pd.DataFrame] = {}
        wandb_log_data = {}

        try:
            # 1. Load and prepare data
            gt_by_scenario, pred_by_scenario = self._load_and_prepare_data(
                benchmark_json_path, retrieved_results
            )

            # 2. Calculate metrics for each scenario + Overall
            all_scenarios_to_process = self.scenario_names + ["Overall"]
            for scenario in all_scenarios_to_process:
                gt_ids = gt_by_scenario.get(scenario)
                pred_ids = pred_by_scenario.get(scenario)

                if not gt_ids or not pred_ids: # Handles cases where a scenario might have 0 examples
                     logging.warning(f"No data found for scenario '{scenario}'. Skipping calculation.")
                     # Create an empty DataFrame with the correct columns/index for consistency
                     columns = ['Precision', 'Recall', 'F1', 'Perfect Recall', 'Perc_Preds_Less_Than_n']
                     results_tables[scenario] = pd.DataFrame(columns=columns, index=pd.Index(self.n_values, name='@n'))
                     continue


                logging.info(f"Calculating metrics for scenario: {scenario} ({len(gt_ids)} instances)")
                scenario_df = self.calculate_metrics(gt_ids, pred_ids)
                results_tables[scenario] = scenario_df

                # Prepare data for W&B logging
                if enable_wandb:
                    wandb_log_data[f"eval/{scenario}_metrics"] = wandb.Table(dataframe=scenario_df.reset_index())


            # 3. Display results
            if verbose:
                print("\n--- Evaluation Results ---")
                print(f"Total Retrieval Time: {total_seconds:.2f} seconds")
                # W&B logging for time happens separately below if enabled

                for scenario, df in results_tables.items():
                    # --- MODIFIED DISPLAY SECTION ---
                    title = f"--- Scenario: {scenario} ---"
                    if _IPYTHON_DISPLAY_AVAILABLE:
                        # Use HTML for richer title formatting in IPython environments
                        display(HTML(f"<h3>{title}</h3>"))
                    else:
                        # Standard print for titles otherwise
                        print(f"\n{title}")

                    if df.empty:
                        print("No results calculated for this scenario.")
                    else:
                        if _IPYTHON_DISPLAY_AVAILABLE:
                            # Use display() for rich DataFrame rendering (HTML table)
                            # Apply float formatting using pandas styler for consistency
                            display(df.style.format("{:.4f}", subset=pd.IndexSlice[:, df.select_dtypes(include=['float']).columns]))
                        else:
                            # Fallback to plain text print if IPython is not available
                            print(df.to_string(float_format="%.4f"))

            # 4. Log to W&B (if enabled)
            if enable_wandb and wandb_log_data:
                wandb.log(wandb_log_data)
                wandb.log({"eval/Total_Retrieval_Time": total_seconds})
                logging.info("Results logged to Weights & Biases.")

        except Exception as e:
            logging.exception("An error occurred during evaluation.") # Log full traceback
            # Ensure W&B run is finished even if an error occurs mid-process
            if enable_wandb and wandb.run:
                 wandb.finish(exit_code=1) # Indicate failure
            raise # Re-raise the exception after logging/cleanup
        finally:
            # Ensure W&B run is finished cleanly on successful completion
            if enable_wandb and wandb.run:
                wandb.finish()

        return results_tables