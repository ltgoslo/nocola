# coding=utf-8

import argparse
import torch
import torch.nn.functional as F
import gzip
import pickle
import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="/fp/projects01/ec30/davisamu/matias-david-andrei/code_datasets/datasets/nocola/nocola_blimp_anonym.txt", type=str, help="Path to BLiMP.")
#    parser.add_argument("--model_name", default="/fp/projects01/ec30/models/xlm-roberta-base", type=str, help="The initial checkpoint to start training from.")
    parser.add_argument("--model_name", default="vesteinn/ScandiBERT", type=str, help="The initial checkpoint to start training from.")
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()

    return args


def is_right(good, bad, model, tokenizer, device):
    mask_index = tokenizer.mask_token_id
    pad_index = tokenizer.pad_token_id
    cls_index = torch.tensor([tokenizer.cls_token_id])
    sep_index = torch.tensor([tokenizer.sep_token_id])

    good = torch.tensor(good)
    bad = torch.tensor(bad)
    labels = torch.cat([good, bad]).unsqueeze(-1).to(device)

    def prepare(tokens, padding: int):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = False
        return input_ids, attention_mask

    good_input_ids, good_attention_mask = prepare(good, max(0, len(bad) - len(good)))
    bad_input_ids, bad_attention_mask = prepare(bad, max(0, len(good) - len(bad)))

    input_ids = torch.cat([good_input_ids, bad_input_ids], dim=0)
    attention_mask = torch.cat([good_attention_mask, bad_attention_mask], dim=0)

    indices = torch.cat([torch.arange(1, 1 + len(good), device=device), torch.arange(1, 1 + len(bad), device=device)])

    total_score = []

    for b in range(input_ids.size(0) // args.batch_size + 1):
        logits = model(
            input_ids[b * args.batch_size : (b+1) * args.batch_size, :].contiguous(),
            attention_mask[b * args.batch_size : (b+1) * args.batch_size, :].contiguous()
        ).logits

        logits = torch.gather(
            logits,
            dim=1,
            index=indices[b * args.batch_size : (b+1) * args.batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
        ).squeeze(1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels[b * args.batch_size : (b+1) * args.batch_size, :], dim=-1).squeeze(-1)
        total_score.append(log_p)

    total_score = torch.cat(total_score, dim=0)

    good_log_p = total_score[:len(good)].mean()
    bad_log_p = total_score[len(good):].mean()

    return good_log_p > bad_log_p + 1e-8


from collections import Counter

@torch.no_grad()
def evaluate(model, tokenizer, pairs, device):
    correct, total = Counter(), Counter()
    pairs_tqdm = tqdm.tqdm(pairs)
    for pair in pairs_tqdm:
        good, bad = pair["good"], pair["bad"]

        good = tokenizer(good, add_special_tokens=False).input_ids
        bad = tokenizer(bad, add_special_tokens=False).input_ids

        if is_right(good, bad, model, tokenizer, device):
            correct["BLiMP/all"] += 1
            correct[f"BLiMP/raw/{pair['error_type']}"] += 1
        total["BLiMP/all"] += 1
        total[f"BLiMP/raw/{pair['error_type']}"] += 1

        pairs_tqdm.set_postfix_str(f"accuracy: {correct['BLiMP/all'] / total['BLiMP/all'] * 100.0:.2f}")

    correct["BLiMP/fine/inflection"] = correct[f"BLiMP/raw/F"] + correct[f"BLiMP/raw/INFL"]
    correct["BLiMP/fine/word choice"] = correct[f"BLiMP/raw/W"] + correct[f"BLiMP/raw/FL"]
    correct["BLiMP/fine/spelling"] = correct[f"BLiMP/raw/ORT"]
    correct["BLiMP/fine/missing"] = correct[f"BLiMP/raw/M"]
    correct["BLiMP/fine/superfluous"] = correct[f"BLiMP/raw/R"]
    correct["BLiMP/fine/punctuation"] = correct[f"BLiMP/raw/PUNC"] + correct[f"BLiMP/raw/PUNCM"] + correct[f"BLiMP/raw/PUNCR"]
    correct["BLiMP/fine/order"] = correct[f"BLiMP/raw/O"]
    correct["BLiMP/fine/case"] = correct[f"BLiMP/raw/CAP"]
    correct["BLiMP/fine/compounding"] = correct[f"BLiMP/raw/PART"] + correct[f"BLiMP/raw/SPL"]
    correct["BLiMP/fine/derivation"] = correct[f"BLiMP/raw/DER"]

    total["BLiMP/fine/inflection"] = total[f"BLiMP/raw/F"] + total[f"BLiMP/raw/INFL"]
    total["BLiMP/fine/word choice"] = total[f"BLiMP/raw/W"] + total[f"BLiMP/raw/FL"]
    total["BLiMP/fine/spelling"] = total[f"BLiMP/raw/ORT"]
    total["BLiMP/fine/missing"] = total[f"BLiMP/raw/M"]
    total["BLiMP/fine/superfluous"] = total[f"BLiMP/raw/R"]
    total["BLiMP/fine/punctuation"] = total[f"BLiMP/raw/PUNC"] + total[f"BLiMP/raw/PUNCM"] + total[f"BLiMP/raw/PUNCR"]
    total["BLiMP/fine/order"] = total[f"BLiMP/raw/O"]
    total["BLiMP/fine/case"] = total[f"BLiMP/raw/CAP"]
    total["BLiMP/fine/compounding"] = total[f"BLiMP/raw/PART"] + total[f"BLiMP/raw/SPL"]
    total["BLiMP/fine/derivation"] = total[f"BLiMP/raw/DER"]

    results = {key: correct[key] / total[key] * 100.0 for key in correct.keys()}
    return results


if __name__ == "__main__":
    args = parse_arguments()

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    if wandb.run is None:
        wandb.init(
            name=f"BLiMP_{args.model_name.split('/')[-1]}",
            project="nocola",
            entity="ltg",
        )

    sentence_pairs = []
    for line in open(args.input_path, "r"):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue

        sentence_1, sentence_2, error_type = line.split('\t')
        if sentence_1 == sentence_2:
            continue

        sentence_pairs.append({
            "good": sentence_2,
            "bad": sentence_1,
            "error_type": error_type
        })

    results = evaluate(model, tokenizer, sentence_pairs, device)

    print("###")
    print(f"{args.model_name.split('/')[-1]} & {results['BLiMP/fine/inflection']:.2f} & {results['BLiMP/fine/word choice']:.2f} & {results['BLiMP/fine/spelling']:.2f}  & {results['BLiMP/fine/missing']:.2f} & {results['BLiMP/fine/superfluous']:.2f} & {results['BLiMP/fine/punctuation']:.2f} & {results['BLiMP/fine/order']:.2f} & {results['BLiMP/fine/case']:.2f} & {results['BLiMP/fine/compounding']:.2f} & {results['BLiMP/fine/derivation']:.2f}")
    print("###", flush=True)

    wandb.log(results)
