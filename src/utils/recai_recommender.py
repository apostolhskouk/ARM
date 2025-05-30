from typing import List, Dict,Callable
import torch
import math

class Node:
    def __init__(self, token_id: int):
        self.token_id: int = token_id
        self.children: Dict[int, Node] = {}
        self.count: int = 0 

    def add_child(self, child_token_id: int):
        if child_token_id not in self.children:
            self.children[child_token_id] = Node(child_token_id)
        self.children[child_token_id].count += 1

    def __getitem__(self, child_token_id: int):
        if child_token_id not in self.children:
            return None
        return self.children[child_token_id]

class Trie_link:
    def __init__(self, item_ids: List[List[int]], tokenizer):
        self.tokenizer = tokenizer
        self.trie: Node = Node(-1) 

        if not hasattr(tokenizer, 'soi_token_id') or not hasattr(tokenizer, 'eoi_token_id'):
            raise ValueError("Tokenizer must have 'soi_token_id' and 'eoi_token_id' attributes.")
        
        self.soi_token_id_val: int = tokenizer.soi_token_id
        self.eoi_token_id_val: int = tokenizer.eoi_token_id

        for token_ids_for_item in item_ids:
            temp_node = self.trie
            token_ids_wcs = [self.soi_token_id_val] + token_ids_for_item + [self.eoi_token_id_val]
            for child_token_id in token_ids_wcs:
                temp_node.add_child(child_token_id)
                temp_node = temp_node[child_token_id]
    
    @property
    def soi_token_id(self) -> int:
        return self.soi_token_id_val

    @property
    def eoi_token_id(self) -> int:
        return self.eoi_token_id_val

    def constrain_search_list(self, batch_id: int, input_ids: torch.Tensor) -> List[int]:
        main_eos_token_id = self.tokenizer.eos_token_id
        if main_eos_token_id is None and hasattr(self.tokenizer, 'eot_token_id'): 
             main_eos_token_id = self.tokenizer.eot_token_id

        last_eos_indices = torch.eq(input_ids, main_eos_token_id).nonzero(as_tuple=True)[0] if main_eos_token_id is not None else torch.tensor([], dtype=torch.long, device=input_ids.device)
        last_eos_index = last_eos_indices[-1].item() if last_eos_indices.numel() > 0 else -1

        soi_indices = torch.eq(input_ids, self.soi_token_id).nonzero(as_tuple=True)[0]
        soi_indices = soi_indices[soi_indices > last_eos_index] 

        eoi_indices = torch.eq(input_ids, self.eoi_token_id).nonzero(as_tuple=True)[0]
        eoi_indices = eoi_indices[eoi_indices > last_eos_index]

        last_eoi_index = eoi_indices[-1].item() if eoi_indices.numel() > 0 else -1
        last_soi_index = soi_indices[-1].item() if soi_indices.numel() > 0 else -1

        if last_soi_index <= last_eoi_index: 
            return None 

        current_path_in_trie_tokens = input_ids[last_soi_index:]
        
        temp_node = self.trie
        for token_in_path in current_path_in_trie_tokens:
            temp_node = temp_node[token_in_path.item()]
            if temp_node is None: 
                return [] 

        if temp_node is not None:
            return [
                next_token_id
                for next_token_id in temp_node.children.keys()
                if isinstance(next_token_id, int) 
            ]
        else:
            return [] 

class LogitsProcessor:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

class FastPrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        
        actual_batch_size = input_ids.shape[0] 
        if actual_batch_size % self._num_beams == 0:
            num_original_batch_items = actual_batch_size // self._num_beams
            for i in range(actual_batch_size):
                original_batch_id = i // self._num_beams
                sent = input_ids[i]
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(original_batch_id, sent)
                if prefix_allowed_tokens is None:
                    mask[i] = 0
                else:
                    if prefix_allowed_tokens:
                        mask[i, prefix_allowed_tokens] = 0
        else:
             for i in range(actual_batch_size):
                sent = input_ids[i]
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(i, sent)
                if prefix_allowed_tokens is None:
                    mask[i] = 0
                else:
                    if prefix_allowed_tokens:
                        mask[i, prefix_allowed_tokens] = 0
        return scores + mask