import flair.models
from typing import List, Union
from flair.data import Label, DataPoint
from extended_flair.extended_sentence import Sentence
import torch
import itertools
from collections import Counter
from pathlib import Path
from typing import Union, List, Tuple, Optional

import torch.nn
from torch.utils.data.dataset import Dataset

import flair
from flair.data import DataPoint, Dictionary, SpanLabel
from flair.datasets import DataLoader, SentenceDataset
from flair.training_utils import Result, store_embeddings

class TextClassifier(flair.models.TextClassifier):
    def __init__(self, **kwargs):
        self.max_token = kwargs.pop('max_token')
        self.max_sentence_parts = kwargs.pop('max_sentence_parts')
        self.default_delimiter = kwargs.pop('default_delimiter')

        super().__init__(**kwargs)

    def split_sentence_to_parts(self, sentence: Sentence):
        return Sentence(max_token=self.max_token,
                        max_sentence_parts=self.max_sentence_parts,
                        default_delimiter=self.default_delimiter).copy_part(sentence)


    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "weight_dict": self.weight_dict,
            "max_token": self.max_token,
            "max_sentence_parts": self.max_sentence_parts,
            "default_delimiter": self.default_delimiter
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        label_type = None if "label_type" not in state.keys() else state["label_type"]

        model = TextClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=label_type,
            multi_label=state["multi_label"],
            multi_label_threshold=0.5 if "multi_label_threshold" not in state.keys() else state["multi_label_threshold"],
            loss_weights=weights,
            max_token=state["max_token"],
            max_sentence_parts=state["max_sentence_parts"],
            default_delimiter=state["default_delimiter"]
        )
        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def index_list(
                field: torch.tensor,
                indexes: List[int]
                ):
        start = 0
        end = 0
        index_array = []
        for index in indexes:
            start = end
            end = end + index

            index_array.append(field[start:end])

        return index_array


    def average_scores(
                    self,
                    scores: torch.Tensor,
                    indexes: List[int],
                    token_count: List[int]
                    ):
        averaged_scores = []
        i = 0
        for sentence_part_scores in self.index_list(scores, indexes):
            sum = 0
            average_score = torch.zeros(len(sentence_part_scores[0])).to(flair.device)
            for sentence_part_score in sentence_part_scores:
                sum += token_count[i]
                average_score += sentence_part_score * token_count[i]
                i += 1

            averaged_scores.append(average_score / sum)

        return torch.stack(averaged_scores)


    def forward_pass(self,
                     sentences: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ):

        all_sentence_parts = []
        token_count = []
        indexes = []
        for sentence in sentences:
            sentence_parts = self.split_sentence_to_parts(sentence)

            for sentence_part in sentence_parts:
                for label in sentence.labels:
                    sentence_part.add_label(self.label_type, label)

                all_sentence_parts.append(sentence_part)
                token_count.append(len(sentence_part))

            indexes.append(len(sentence_parts))

        # embed sentences
        self.document_embeddings.embed(all_sentence_parts)

        # make tensor for all embedded sentences in batch
        embedding_names = self.document_embeddings.get_names()
        text_embedding_list = [sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in all_sentence_parts]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        # send through decoder to get logits
        scores = self.decoder(text_embedding_tensor)

        averaged_scores = self.average_scores(scores, indexes, token_count)

        labels = []
        for sentence in sentences:
            labels.append([label.value for label in sentence.get_labels(self.label_type)])

        # minimal return is scores and labels
        return_tuple = (averaged_scores, labels)

        if return_label_candidates:
            label_candidates = [Label(value=None) for sentence in sentences]
            return_tuple += (sentences, label_candidates)

        return return_tuple

    def evaluate(
                self,
                data_points: Union[List[DataPoint], Dataset],
                gold_label_type: str, out_path: Union[str, Path] = None,
                embedding_storage_mode: str = "none",
                mini_batch_size: int = 32,
                num_workers: int = 8,
                main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
                exclude_labels: List[str] = [],
                gold_label_dictionary: Optional[Dictionary] = None
    )-> Result:
        import numpy as np
        import sklearn

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(data_points, Dataset):
            data_points = SentenceDataset(data_points)

        data_loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=num_workers)

        with torch.no_grad():

            # loss calculation
            eval_loss = 0
            average_over = 0

            # variables for printing
            lines: List[str] = []
            is_word_level = False

            # variables for computing scores
            all_spans: List[str] = []
            all_true_values = {}
            all_predicted_values = {}

            sentence_id = 0
            for batch in data_loader:

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels('predicted')

                # predict for batch
                self.predict(batch,
                            embedding_storage_mode=embedding_storage_mode,
                            mini_batch_size=mini_batch_size,
                            label_name='predicted',
                            return_loss=True)

                # get the gold labels
                for datapoint in batch:

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ': ' + gold_label.identifier

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = '<unk>'

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.append(representation)

                        if type(gold_label) == SpanLabel: is_word_level = True

                    for predicted_span in datapoint.get_labels("predicted"):
                        representation = str(sentence_id) + ': ' + predicted_span.identifier

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                        else:
                            all_predicted_values[representation].append(predicted_span.value)

                        if representation not in all_spans:
                            all_spans.append(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    for datapoint in batch:

                        # if the model is span-level, transfer to word-level annotations for printout
                        if is_word_level:

                            # all labels default to "O"
                            for token in datapoint:
                                token.set_label("gold_bio", "O")
                                token.set_label("predicted_bio", "O")

                            # set gold token-level
                            for gold_label in datapoint.get_labels(gold_label_type):
                                gold_label: SpanLabel = gold_label
                                prefix = "B-"
                                for token in gold_label.span:
                                    token.set_label("gold_bio", prefix + gold_label.value)
                                    prefix = "I-"

                            # set predicted token-level
                            for predicted_label in datapoint.get_labels("predicted"):
                                predicted_label: SpanLabel = predicted_label
                                prefix = "B-"
                                for token in predicted_label.span:
                                    token.set_label("predicted_bio", prefix + predicted_label.value)
                                    prefix = "I-"

                            # now print labels in CoNLL format
                            for token in datapoint:
                                eval_line = f"{token.text} " \
                                            f"{token.get_tag('gold_bio').value} " \
                                            f"{token.get_tag('predicted_bio').value}\n"
                                lines.append(eval_line)
                            lines.append("\n")
                        else:
                            # check if there is a label mismatch
                            g = [label.identifier + label.value for label in datapoint.get_labels(gold_label_type)]
                            p = [label.identifier + label.value for label in datapoint.get_labels('predicted')]
                            g.sort()
                            p.sort()
                            correct_string = " -> MISMATCH!\n" if g != p else ""
                            # print info
                            eval_line = f"{datapoint.to_original_text()}\n" \
                                        f" - Gold: {datapoint.get_labels(gold_label_type)}\n" \
                                        f" - Pred: {datapoint.get_labels('predicted')}\n{correct_string}\n"
                            lines.append(eval_line)

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

            # finally, compute numbers
            y_true = []
            y_pred = []

            for span in all_spans:

                true_values = all_true_values[span] if span in all_true_values else ['O']
                predicted_values = all_predicted_values[span] if span in all_predicted_values else ['O']

                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_values:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())

        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter()
        counter.update(list(itertools.chain.from_iterable(all_true_values.values())))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == 'O': continue
            if label_name in exclude_labels: continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True, labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)

            precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
            recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
            micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! Could be an error in your corpus or how you "
                "initialize the trainer!")
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {micro_f_score}"
                f"\n- F-score (macro) {macro_f_score}"
                f"\n- Accuracy {accuracy_score}"
                "\n\nBy class:\n" + classification_report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss
        )

        return result