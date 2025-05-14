import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from src.retrieval.base import RetrievalResult
from src.utils.evaluator import EvaluationMetrics
import re
class FeverousEvaluation(EvaluationMetrics):
    """
    Evaluates retrieval performance on the FEVEROUS dataset. Handles various evidence types.
    Levels:
    1. Exact: Exac  t sentence ID (page_sentence_X) or Table ID (page_table_Y).
    2. Table+Page: Table ID (page_table_Y) or Page ID (page) for sentences.
    3. Page: Page ID (page) only.
    Ignores 'item' type evidence.
    """
    def __init__(self, n_values: List[int]):
        super().__init__(n_values)
        self.results_exact_sentence_table: Optional[pd.DataFrame] = None
        self.results_table_page: Optional[pd.DataFrame] = None
        self.results_page: Optional[pd.DataFrame] = None
        self.table_index_regex = re.compile(
            r"_(?:cell|header_cell|table_caption)_(\d+).*"
        )


    def _get_representative_gt_id(self, evidence_id: str) -> Optional[Tuple[str, str]]:
        """
        Parses a FEVEROUS evidence ID and returns the page and its representative ID
        (sentence_X or table_Y), or None if the type should be ignored (e.g., item).
        """
        parts = evidence_id.split('_', 1)
        if len(parts) < 2:
             print(f"Warning: Could not parse evidence ID: {evidence_id}")
             return None # Cannot determine page and id_part

        page = parts[0]
        id_part = parts[1]

        if id_part.startswith('sentence_'):
            return page, id_part
        elif id_part.startswith('cell_') or \
             id_part.startswith('header_cell_') or \
             id_part.startswith('table_caption_'):
            # Format: cell_X_Y_Z, header_cell_X_Y_Z, table_caption_X
            match = self.table_index_regex.match('_' + id_part) # Add underscore back
            if match:
                table_index = match.group(1)
                return page, f"table_{table_index}"
            else:
                print(f"Warning: Could not extract table index from table evidence: {evidence_id}")
                return page, "table_unknown" # Fallback if regex fails
        elif id_part.startswith('table_') and not id_part.startswith('table_caption_'):
            # Direct table format: table_X (less common in GT?)
             return page, id_part
        elif id_part.startswith('item_'):
            # Ignore item types
            return None
        else:
            return None # Ignore other unknown formats

    def _get_ground_truth_lists(self, feverous_data: List[Dict]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """ Parses the FEVEROUS data to create ground truth lists for each granularity. """
        gt_exact_list: List[List[str]] = []
        gt_table_page_list: List[List[str]] = []
        gt_page_list: List[List[str]] = []

        for record in feverous_data:
            instance_gt_exact: Set[str] = set()
            instance_gt_table_page: Set[str] = set()
            instance_gt_page: Set[str] = set()

            evidence_ids = record.get("evidence_ids", [])

            for ev_id in evidence_ids:
                parse_result = self._get_representative_gt_id(ev_id)
                if parse_result is None:
                    continue # Ignore this evidence ID (e.g., it was an 'item')

                page, representative_id = parse_result

                # 1. Exact Sentence / Table ID Level
                instance_gt_exact.add(f"{page}_{representative_id}")

                # 2. Table ID (Exact) + Page ID (for Sentences) Level
                if representative_id.startswith("table_"):
                     instance_gt_table_page.add(f"{page}_{representative_id}")
                elif representative_id.startswith("sentence_"):
                     instance_gt_table_page.add(page)

                # 3. Page Level
                instance_gt_page.add(page)

            gt_exact_list.append(list(instance_gt_exact))
            gt_table_page_list.append(list(instance_gt_table_page))
            gt_page_list.append(list(instance_gt_page))

        return gt_exact_list, gt_table_page_list, gt_page_list

    def _get_prediction_lists(self, predictions: List[List[RetrievalResult]]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """ Parses the RetrievalResult predictions for each granularity. """
        pred_exact_list: List[List[str]] = []
        pred_table_page_list: List[List[str]] = []
        pred_page_list: List[List[str]] = []

        for instance_predictions in predictions:
            instance_pred_exact = []
            instance_pred_table_page = []
            instance_pred_page = []

            for result in instance_predictions:
                metadata = result.metadata
                page_title = metadata.get('page_title')
                source = metadata.get('source') # e.g., "sentence_9", "table_0"

                if not page_title or source is None:
                    # print(f"Warning: Skipping prediction due to missing metadata: {result}")
                    continue

                if not (source.startswith("sentence_") or source.startswith("table_")):
                    # print(f"Warning: Skipping prediction due to unexpected source format: {source} in {result}")
                    continue

                # 1. Exact Sentence / Table ID Level Prediction
                exact_pred_id = f"{page_title}_{source}"
                instance_pred_exact.append(exact_pred_id)

                # 2. Table ID (Exact) + Page ID (for Sentences) Level Prediction
                if source.startswith("table_"):
                    table_page_pred_id = exact_pred_id
                elif source.startswith("sentence_"):
                    table_page_pred_id = page_title
                else:
                     continue
                instance_pred_table_page.append(table_page_pred_id)

                # 3. Page Level Prediction
                instance_pred_page.append(page_title)

            pred_exact_list.append(instance_pred_exact)
            pred_table_page_list.append(instance_pred_table_page)
            pred_page_list.append(instance_pred_page)

        return pred_exact_list, pred_table_page_list, pred_page_list

    def evaluate(self, json_path: str, predictions: List[List[RetrievalResult]]) -> Dict[str, pd.DataFrame]:
        """
        Performs FEVEROUS evaluation at three granularity levels.

        Args:
            json_path: Path to the FEVEROUS JSON data file (expected as a single JSON array).
            predictions: A list of lists, where each inner list contains RetrievalResult
                         objects for the corresponding claim in the JSON file.

        Returns:
            A dictionary containing the evaluation DataFrames for 'exact_sentence_table',
            'table_page', and 'page' levels.
        """
        data_path = Path(json_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"JSON file not found at: {json_path}")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                # Changed from json.loads(line) to json.load(f) for single JSON array format
                feverous_data = json.load(f)
        except json.JSONDecodeError as e:
            # Provide more context for JSON decoding errors
            raise ValueError(f"Error decoding JSON file '{json_path}': {e}. Ensure it's a valid JSON array.") from e
        except Exception as e:
            raise RuntimeError(f"Error reading file {json_path}: {e}")

        # Validate that feverous_data is a list
        if not isinstance(feverous_data, list):
             raise ValueError(f"Error: Expected a JSON list (array) in {json_path}, but got type {type(feverous_data)}.")


        if len(feverous_data) != len(predictions):
            raise ValueError(
                f"Number of records in JSON ({len(feverous_data)}) does not match "
                f"number of prediction lists ({len(predictions)})."
            )

        # Keep print statements minimal as requested previously
        # print("Parsing ground truth...")
        gt_exact, gt_table_page, gt_page = self._get_ground_truth_lists(feverous_data)

        # print("Parsing predictions...")
        pred_exact, pred_table_page, pred_page = self._get_prediction_lists(predictions)

        # print("\nCalculating 'Exact Sentence / Table ID' metrics...")
        self.results_exact_sentence_table = self.calculate_metrics(gt_exact, pred_exact)
        # print(self.results_exact_sentence_table)

        # print("\nCalculating 'Table ID + Page ID (for Sentence)' metrics...")
        self.results_table_page = self.calculate_metrics(gt_table_page, pred_table_page)
        # print(self.results_table_page)

        # print("\nCalculating 'Page ID' metrics...")
        self.results_page = self.calculate_metrics(gt_page, pred_page)
        # print(self.results_page)

        return {
            "exact_sentence_table": self.results_exact_sentence_table,
            "table_page": self.results_table_page,
            "page": self.results_page
        }

    def visualize_all_results(self):
        """Visualizes the results for all three evaluation levels."""
        if self.results_exact_sentence_table is not None:
            self.visualize_results(df_to_plot=self.results_exact_sentence_table, title="FEVEROUS Performance (Exact Sentence/Table)")
        else:
            print("No 'Exact Sentence/Table' results to visualize.")

        if self.results_table_page is not None:
            self.visualize_results(df_to_plot=self.results_table_page, title="FEVEROUS Performance (Table ID + Page for Sentence)")
        else:
            print("No 'Table+Page' results to visualize.")

        if self.results_page is not None:
            self.visualize_results(df_to_plot=self.results_page, title="FEVEROUS Performance (Page Level)")
        else:
            print("No 'Page' results to visualize.")
